
use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Size, Rect, Point, RNG, Vec3b, Scalar, Vector},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter6/";

pub(crate) fn run() -> Result<()> {
    // 构建建议矩阵，用于求取像素之间的距离
    let mut a = Mat::zeros_size(Size::new(5, 5), opencv::core::CV_8U)?.to_mat()?;
    {
        let data = vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 0, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ];
        a.data_typed_mut::<u8>()?.copy_from_slice(&data);
    }
    let mut dist_l1 = Mat::default();
    let mut dist_l2 = Mat::default();
    let mut dist_c = Mat::default();
    // 计算街区距离
    imgproc::distance_transform(&a, &mut dist_l1, imgproc::DIST_L1, 3,  opencv::core::CV_8U)?;
    println!("街区距离：{:?}", dist_l1);

    // 计算欧式距离
    imgproc::distance_transform(&a, &mut dist_l2, imgproc::DIST_L2, 5, opencv::core::CV_8U)?;
    println!("欧式距离：{:?}", dist_l2);

    // 计算棋盘距离
    imgproc::distance_transform(&a, &mut dist_c, imgproc::DIST_C, 5, opencv::core::CV_8U)?;
    println!("棋盘距离：{:?}", dist_c);

    // 对图像进行距离变换
    let rice = imgcodecs::imread(&format!("{}{}", BASE_PATH, "rice.png"), imgcodecs::IMREAD_GRAYSCALE)?;
    if rice.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }
    let mut rice_bw = Mat::default();
    let mut rice_bw_inv = Mat::default();

    // 将图像转成二值图像，同时把黑白区域颜色反转
    imgproc::threshold(&rice, &mut rice_bw, 50.0, 255.0, imgproc::THRESH_BINARY)?;
    imgproc::threshold(&rice, &mut rice_bw_inv, 50.0, 255.0, imgproc::THRESH_BINARY_INV)?;

    // 距离变换
    let mut dist = Mat::default();
    let mut dist_inv = Mat::default();
    imgproc::distance_transform(&rice_bw, &mut dist, 1, 3, opencv::core::CV_32F)?;  // 为了显示清晰，将数据类型变成 CV_32F
    imgproc::distance_transform(&rice_bw_inv, &mut dist_inv, 1, 3, opencv::core::CV_8U)?;

    // 显示变换结果
    highgui::imshow("riceBW", &rice_bw)?;
    highgui::imshow("dist", &dist)?;
    highgui::imshow("riceBW_INV", &rice_bw_inv)?;
    highgui::imshow("dist_INV", &dist_inv)?;

    highgui::wait_key(0)?;

    Ok(())
}
