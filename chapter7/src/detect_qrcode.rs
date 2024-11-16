use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};
use zbar_rust::{ZBarConfig, ZBarImage, ZBarImageScanner, ZBarSymbolType};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&(BASE_PATH.to_owned() + "qrcode2.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 转换为灰度图
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // 创建 ZBar 扫描器
    let mut scanner = ZBarImageScanner::new();
    scanner.set_config(ZBarSymbolType::ZBarNone, ZBarConfig::ZBarCfgEnable, 1).unwrap();

    // 创建 ZBar 图像对象
    let  symbols = scanner.scan_y800(gray.data_bytes()?, gray.cols() as u32, gray.rows() as u32).unwrap();
    // 检查是否检测到条码
    if symbols.is_empty() {
        println!("no barcode detected");
    } else {
        for symbol in symbols {
            println!("decoded symbol: {:?}", symbol);
        }
    }

    // 显示图像
    highgui::imshow("img", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}