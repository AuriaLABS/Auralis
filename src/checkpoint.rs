//! Checkpoint AURLIS01/02/03: config + tokenizer + params + Adam opcional.
use crate::bpe::BpeTokenizer;
use crate::model::{Config, Gpt};
use crate::optim::Adam;
use crate::tokenizer::{AnyTok, CharTokenizer};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC1: &[u8; 8] = b"AURLIS01";
const MAGIC2: &[u8; 8] = b"AURLIS02";
const MAGIC3: &[u8; 8] = b"AURLIS03";

fn write_u32(f: &mut File, n: u32) -> std::io::Result<()> {
    f.write_all(&n.to_le_bytes())
}
fn read_u32(f: &mut File) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn write_str(f: &mut File, s: &str) -> std::io::Result<()> {
    let b = s.as_bytes();
    write_u32(f, b.len() as u32)?;
    f.write_all(b)
}
fn read_str(f: &mut File) -> std::io::Result<String> {
    let n = read_u32(f)? as usize;
    let mut b = vec![0u8; n];
    f.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

pub fn save(path: impl AsRef<Path>, gpt: &Gpt, tok: &AnyTok) -> std::io::Result<()> {
    save_full(path, gpt, tok, None)
}

pub fn save_full(
    path: impl AsRef<Path>,
    gpt: &Gpt,
    tok: &AnyTok,
    adam: Option<&Adam>,
) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(if adam.is_some() { MAGIC3 } else { MAGIC2 })?;
    let cfg = gpt.cfg;
    for n in [cfg.vocab as u32, cfg.n_embd as u32, cfg.n_head as u32, cfg.n_layer as u32, cfg.block as u32, cfg.n_ff as u32] {
        write_u32(&mut f, n)?;
    }
    match tok {
        AnyTok::Char(t) => {
            write_u32(&mut f, 0)?;
            write_str(&mut f, &t.itos.iter().collect::<String>())?;
        }
        AnyTok::Bpe(t) => {
            write_u32(&mut f, 1)?;
            write_u32(&mut f, t.itos.len() as u32)?;
            for s in &t.itos { write_str(&mut f, s)?; }
            write_u32(&mut f, t.merges.len() as u32)?;
            for (a, b) in &t.merges {
                write_str(&mut f, a)?;
                write_str(&mut f, b)?;
            }
        }
    }
    let params = gpt.collect_params();
    write_u32(&mut f, params.len() as u32)?;
    for x in params { f.write_all(&x.to_le_bytes())?; }
    if let Some(adam) = adam {
        let (lr, t, m, v) = adam.export();
        f.write_all(&lr.to_le_bytes())?;
        f.write_all(&t.to_le_bytes())?;
        write_u32(&mut f, m.len() as u32)?;
        for x in m { f.write_all(&x.to_le_bytes())?; }
        for x in v { f.write_all(&x.to_le_bytes())?; }
    }
    Ok(())
}

pub fn load(path: impl AsRef<Path>) -> std::io::Result<(Gpt, AnyTok)> {
    load_full(path).map(|(g, t, _)| (g, t))
}

pub fn load_full(path: impl AsRef<Path>) -> std::io::Result<(Gpt, AnyTok, Option<Adam>)> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic == MAGIC1 {
        return load_v1(&mut f).map(|(g, t)| (g, t, None));
    }
    let with_adam = &magic == MAGIC3;
    if &magic != MAGIC2 && !with_adam {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "no es un checkpoint Auralis"));
    }
    let vocab_n = read_u32(&mut f)? as usize;
    let cfg = Config {
        vocab: vocab_n,
        n_embd: read_u32(&mut f)? as usize,
        n_head: read_u32(&mut f)? as usize,
        n_layer: read_u32(&mut f)? as usize,
        block: read_u32(&mut f)? as usize,
        n_ff: read_u32(&mut f)? as usize,
    };
    let kind = read_u32(&mut f)?;
    let tok = match kind {
        0 => {
            let vocab = read_str(&mut f)?;
            let itos: Vec<char> = vocab.chars().collect();
            let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
            AnyTok::Char(CharTokenizer { stoi, itos })
        }
        1 => {
            let nv = read_u32(&mut f)? as usize;
            let mut itos = Vec::with_capacity(nv);
            for _ in 0..nv { itos.push(read_str(&mut f)?); }
            let nm = read_u32(&mut f)? as usize;
            let mut merges = Vec::with_capacity(nm);
            for _ in 0..nm { merges.push((read_str(&mut f)?, read_str(&mut f)?)); }
            let stoi = itos.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
            AnyTok::Bpe(BpeTokenizer { itos, stoi, merges })
        }
        _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "tokenizer desconocido")),
    };
    if tok.vocab_size() != cfg.vocab {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "vocab size mismatch"));
    }
    let n = read_u32(&mut f)? as usize;
    let mut raw = vec![0u8; n * 4];
    f.read_exact(&mut raw)?;
    let mut params = Vec::with_capacity(n);
    for chunk in raw.chunks_exact(4) {
        params.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    let mut rng = rand::thread_rng();
    let mut gpt = Gpt::new(cfg, &mut rng);
    gpt.write_params(&params);
    let adam = if with_adam {
        let mut b4 = [0u8; 4];
        f.read_exact(&mut b4)?;
        let lr = f32::from_le_bytes(b4);
        f.read_exact(&mut b4)?;
        let t = i32::from_le_bytes(b4);
        let nm = read_u32(&mut f)? as usize;
        let mut mraw = vec![0u8; nm * 4];
        f.read_exact(&mut mraw)?;
        let mut vraw = vec![0u8; nm * 4];
        f.read_exact(&mut vraw)?;
        let m = mraw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let v = vraw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        Some(Adam::from_state(lr, t, m, v))
    } else {
        None
    };
    Ok((gpt, tok, adam))
}

fn load_v1(f: &mut File) -> std::io::Result<(Gpt, AnyTok)> {
    let vocab_n = read_u32(f)? as usize;
    let cfg = Config {
        vocab: vocab_n,
        n_embd: read_u32(f)? as usize,
        n_head: read_u32(f)? as usize,
        n_layer: read_u32(f)? as usize,
        block: read_u32(f)? as usize,
        n_ff: read_u32(f)? as usize,
    };
    let vlen = read_u32(f)? as usize;
    let mut vb = vec![0u8; vlen];
    f.read_exact(&mut vb)?;
    let vocab = String::from_utf8(vb).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let itos: Vec<char> = vocab.chars().collect();
    let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let tok = AnyTok::Char(CharTokenizer { stoi, itos });
    let n = read_u32(f)? as usize;
    let mut raw = vec![0u8; n * 4];
    f.read_exact(&mut raw)?;
    let mut params = Vec::with_capacity(n);
    for chunk in raw.chunks_exact(4) {
        params.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    let mut rng = rand::thread_rng();
    let mut gpt = Gpt::new(cfg, &mut rng);
    gpt.write_params(&params);
    Ok((gpt, tok))
}
