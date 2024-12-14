fn main() {
    let frame = {
        let mut f = vec![];
        ureq::get(&std::env::var("FRAME_URL").unwrap())
            .call()
            .unwrap()
            .into_reader()
            .read_to_end(&mut f)
            .unwrap();
        f
    };
    dbg!(frame);
}
