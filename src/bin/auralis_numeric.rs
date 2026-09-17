fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match auralis::numeric::cli(&args) {
        Ok(out) => {
            print!("{}", out.text);
            if !out.finite {
                std::process::exit(2);
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}
