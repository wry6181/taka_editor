use image::ImageReader;
use std::io::Cursor;

pub struct TakaImage {
  pub color: Vec<[u8; 4]>,
  pub position: Vec<(u32, u32)>,
}

pub fn load_image_from_bytes(bytes: &[u8]) -> Result<TakaImage, Box<dyn std::error::Error>> {
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?
        .to_rgba8();

    let (colors, positions): (Vec<[u8; 4]>, Vec<(u32, u32)>) = img
        .enumerate_pixels()
        .map(|(x, y, p)| (p.0, (x, y)))
        .unzip();

    Ok(TakaImage { color: (colors), position: (positions) })
}
