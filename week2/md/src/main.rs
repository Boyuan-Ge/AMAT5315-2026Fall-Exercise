fn main() {
    if let Err(error) = md::run_cli() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
