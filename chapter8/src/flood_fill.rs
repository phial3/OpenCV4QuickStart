use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Rect, Vec2i, Vector, RNG},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter8/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let mut img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    // 创建随机数生成器
    let mut rng = RNG::default()?;
    rng.set_state(10086);

    // 设置操作标志flags
    let connectivity = 4; // 连通邻域方式
    let mask_val = 255; // 掩码图像的数值
    let flags = connectivity | (mask_val << 8) | imgproc::FLOODFILL_FIXED_RANGE; // 漫水填充操作方式标志

    // 设置与选中像素点的差值
    let lo_diff = Scalar::new(20.0, 20.0, 20.0, 0.0);
    let up_diff = Scalar::new(20.0, 20.0, 20.0, 0.0);

    // 声明掩模矩阵变量
    let mut mask = Mat::zeros(
        img.rows() + 2,
        img.cols() + 2,
        opencv::core::CV_8UC1
    )?.to_mat()?;

    loop {
        // 随机产生图像中某一像素点
        let py = rng.uniform(0, img.rows() - 1)?;
        let px = rng.uniform(0, img.cols() - 1)?;
        let point = Point::new(px, py);

        // 彩色图像中填充的像素值
        let new_val = Scalar::new(
            rng.uniform(0, 255)? as f64,
            rng.uniform(0, 255)? as f64,
            rng.uniform(0, 255)? as f64,
            0.0
        );

        // 漫水填充函数
        let mut rect = Rect::default();
        let area = imgproc::flood_fill_mask(
            &mut img,
            &mut mask,
            point,
            new_val,
            &mut rect,
            lo_diff,
            up_diff,
            flags
        )?;

        // 输出像素点和填充的像素数目
        println!("像素点x：{}  y:{}     填充像素数目：{}", point.x, point.y, area);

        // 输出填充的图像结果
        highgui::imshow("填充的彩色图像", &img)?;
        highgui::imshow("掩模图像", &mask)?;

        // 判断是否结束程序
        let key = highgui::wait_key(0)?;
        if (key & 0xFF) == 27 {
            break;
        }
    }

    Ok(())
}

