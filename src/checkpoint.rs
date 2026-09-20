//! Checkpoint AURLIS01/02/03: config + tokenizer + parameters + optional Adam.
//!
//! Loading untrusted or obsolete checkpoints must return `InvalidData`; it must
//! never reach a parameter-size assertion inside the model.

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
const MAX_CHECKPOINT_STRING: u32 = 1_048_576;
const MAX_BPE_TABLE: u32 = 65_536;

fn invalid_data(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into())
}

fn invalid_input(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, msg.into())
}

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
    let n = u32::try_from(b.len()).map_err(|_| invalid_input("string too large for checkpoint"))?;
    write_u32(f, n)?;
    f.write_all(b)
}

fn read_str(f: &mut File) -> std::io::Result<String> {
    let n = read_u32(f)?;
    if n > MAX_CHECKPOINT_STRING {
        return Err(invalid_data(format!(
            "checkpoint string length {n} exceeds {MAX_CHECKPOINT_STRING}"
        )));
    }
    let mut b = vec![0u8; n as usize];
    f.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| invalid_data(e.to_string()))
}

fn read_bounded_count(f: &mut File, what: &str) -> std::io::Result<u32> {
    let n = read_u32(f)?;
    if n > MAX_BPE_TABLE {
        return Err(invalid_data(format!(
            "{what} count {n} exceeds {MAX_BPE_TABLE}"
        )));
    }
    Ok(n)
}

fn validate_cfg(cfg: Config) -> std::io::Result<()> {
    if cfg.vocab <= 1
        || cfg.n_embd == 0
        || cfg.n_head == 0
        || cfg.n_layer == 0
        || cfg.block == 0
        || cfg.n_ff == 0
        || cfg.n_embd % cfg.n_head != 0
        || cfg.vocab > 65_536
        || cfg.n_embd > 4_096
        || cfg.n_head > 256
        || cfg.n_layer > 64
        || cfg.block > 8_192
        || cfg.n_ff > 16_384
    {
        return Err(invalid_data("invalid model configuration in checkpoint"));
    }
    Ok(())
}

fn build_model(cfg: Config) -> std::io::Result<Gpt> {
    validate_cfg(cfg)?;
    let mut rng = rand::thread_rng();
    Ok(Gpt::new(cfg, &mut rng))
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
    if tok.vocab_size() != gpt.cfg.vocab {
        return Err(invalid_input("tokenizer/model vocabulary mismatch"));
    }

    let params = gpt.collect_params();
    let n_params =
        u32::try_from(params.len()).map_err(|_| invalid_input("model too large for AURLIS03"))?;

    if let Some(adam) = adam {
        let (_, _, m, v) = adam.export();
        if m.len() != params.len() || v.len() != params.len() {
            return Err(invalid_input("Adam state/model parameter mismatch"));
        }
    }

    let mut f = File::create(path)?;
    f.write_all(if adam.is_some() { MAGIC3 } else { MAGIC2 })?;

    let cfg = gpt.cfg;
    for n in [
        cfg.vocab,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_layer,
        cfg.block,
        cfg.n_ff,
    ] {
        write_u32(
            &mut f,
            u32::try_from(n).map_err(|_| invalid_input("model dimension exceeds checkpoint format"))?,
        )?;
    }

    match tok {
        AnyTok::Char(t) => {
            write_u32(&mut f, 0)?;
            write_str(&mut f, &t.itos.iter().collect::<String>())?;
        }
        AnyTok::Bpe(t) => {
            write_u32(&mut f, 1)?;
            write_u32(
                &mut f,
                u32::try_from(t.itos.len())
                    .map_err(|_| invalid_input("BPE vocabulary too large"))?,
            )?;
            for s in &t.itos {
                write_str(&mut f, s)?;
            }
            write_u32(
                &mut f,
                u32::try_from(t.merges.len())
                    .map_err(|_| invalid_input("too many BPE merges"))?,
            )?;
            for (a, b) in &t.merges {
                write_str(&mut f, a)?;
                write_str(&mut f, b)?;
            }
        }
    }

    write_u32(&mut f, n_params)?;
    for x in params {
        f.write_all(&x.to_le_bytes())?;
    }

    if let Some(adam) = adam {
        let (lr, t, m, v) = adam.export();
        f.write_all(&lr.to_le_bytes())?;
        f.write_all(&t.to_le_bytes())?;
        write_u32(
            &mut f,
            u32::try_from(m.len()).map_err(|_| invalid_input("Adam state too large"))?,
        )?;
        for x in m {
            f.write_all(&x.to_le_bytes())?;
        }
        for x in v {
            f.write_all(&x.to_le_bytes())?;
        }
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
        return Err(invalid_data("not an Auralis checkpoint"));
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
    validate_cfg(cfg)?;

    let kind = read_u32(&mut f)?;
    let tok = match kind {
        0 => {
            let vocab = read_str(&mut f)?;
            let itos: Vec<char> = vocab.chars().collect();
            let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
            AnyTok::Char(CharTokenizer { stoi, itos })
        }
        1 => {
            let nv = read_bounded_count(&mut f, "BPE vocabulary")? as usize;
            let mut itos = Vec::with_capacity(nv);
            for _ in 0..nv {
                itos.push(read_str(&mut f)?);
            }
            let nm = read_bounded_count(&mut f, "BPE merges")? as usize;
            let mut merges = Vec::with_capacity(nm);
            for _ in 0..nm {
                merges.push((read_str(&mut f)?, read_str(&mut f)?));
            }
            let stoi = itos
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, s)| (s, i))
                .collect();
            AnyTok::Bpe(BpeTokenizer { itos, stoi, merges })
        }
        _ => return Err(invalid_data("unknown tokenizer type")),
    };

    if tok.vocab_size() != cfg.vocab {
        return Err(invalid_data("vocabulary size mismatch"));
    }

    let mut gpt = build_model(cfg)?;
    let expected = gpt.collect_params().len();
    let n = read_u32(&mut f)? as usize;
    if n != expected {
        return Err(invalid_data(format!(
            "parameter count mismatch: checkpoint={n}, architecture={expected}"
        )));
    }

    let byte_len = n
        .checked_mul(4)
        .ok_or_else(|| invalid_data("parameter byte count overflow"))?;
    let mut raw = vec![0u8; byte_len];
    f.read_exact(&mut raw)?;
    let params: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    gpt.write_params(&params);

    let adam = if with_adam {
        let mut b4 = [0u8; 4];
        f.read_exact(&mut b4)?;
        let lr = f32::from_le_bytes(b4);
        f.read_exact(&mut b4)?;
        let t = i32::from_le_bytes(b4);
        let nm = read_u32(&mut f)? as usize;
        if nm != expected {
            return Err(invalid_data(format!(
                "Adam state mismatch: checkpoint={nm}, architecture={expected}"
            )));
        }

        let state_bytes = nm
            .checked_mul(4)
            .ok_or_else(|| invalid_data("Adam byte count overflow"))?;
        let mut mraw = vec![0u8; state_bytes];
        f.read_exact(&mut mraw)?;
        let mut vraw = vec![0u8; state_bytes];
        f.read_exact(&mut vraw)?;

        let m = mraw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let v = vraw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
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
    validate_cfg(cfg)?;

    let vlen = read_u32(f)?;
    if vlen > MAX_CHECKPOINT_STRING {
        return Err(invalid_data(format!(
            "AURLIS01 vocab length {vlen} exceeds {MAX_CHECKPOINT_STRING}"
        )));
    }
    let mut vb = vec![0u8; vlen as usize];
    f.read_exact(&mut vb)?;
    let vocab = String::from_utf8(vb).map_err(|e| invalid_data(e.to_string()))?;
    let itos: Vec<char> = vocab.chars().collect();
    let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let tok = AnyTok::Char(CharTokenizer { stoi, itos });

    if tok.vocab_size() != cfg.vocab {
        return Err(invalid_data("vocabulary size mismatch"));
    }

    let mut gpt = build_model(cfg)?;
    let expected = gpt.collect_params().len();
    let n = read_u32(f)? as usize;
    if n != expected {
        return Err(invalid_data(format!(
            "AURLIS01 parameter count mismatch: checkpoint={n}, architecture={expected}"
        )));
    }

    let byte_len = n
        .checked_mul(4)
        .ok_or_else(|| invalid_data("parameter byte count overflow"))?;
    let mut raw = vec![0u8; byte_len];
    f.read_exact(&mut raw)?;
    let params: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    gpt.write_params(&params);

    Ok((gpt, tok))
}
