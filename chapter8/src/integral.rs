use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, RNG},
    highgui,
    imgcodecs,
    imgproc,
    photo,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter8/";

pub(crate) fn run() -> Result<()> {
    // 创建一个16×16全为1的矩阵，因为256=16×16
    let mut img = Mat::ones(16, 16, opencv::core::CV_32FC1)?.to_mat()?;

    // 在图像中加入随机噪声
    let mut rng = RNG::default()?;
    rng.set_state(10086);

    for y in 0..img.rows() {
        for x in 0..img.cols() {
            let d = rng.uniform_f32(-0.5, 0.5)?;
            // 获取并修改像素值
            unsafe {
                let pixel = img.at_2d_mut::<f32>(y, x)?;
                *pixel += d;
            }
        }
    }

    // 计算标准求和积分
    let mut sum = Mat::default();
    imgproc::integral(
        &img,
        &mut sum,
        -1, // depth 参数，-1表示使用默认深度
    )?;
    // 为了便于显示，转成CV_8U格式
    let sum8u = Mat::convert_to_8u(&sum)?;

    // 计算平方求和积分
    let mut sqsum = Mat::default();
    imgproc::integral2(
        &img,
        &mut sum,
        &mut sqsum,
        -1,
        -1,
    )?;
    // 为了便于显示，转成CV_8U格式
    let sqsum8u = Mat::convert_to_8u(&sqsum)?;

    // 计算倾斜求和积分
    let mut tilted = Mat::default();
    imgproc::integral3(
        &img,
        &mut sum,
        &mut sqsum,
        &mut tilted,
        -1,
        -1,
    )?;
    // 为了便于显示，转成CV_8U格式
    let tilted8u = Mat::convert_to_8u(&tilted)?;

    // 创建窗口
    highgui::named_window("sum8U", highgui::WINDOW_NORMAL)?;
    highgui::named_window("sqsum8U", highgui::WINDOW_NORMAL)?;
    highgui::named_window("tilted8U", highgui::WINDOW_NORMAL)?;

    // 显示结果
    highgui::imshow("sum8U", &sum8u)?;
    highgui::imshow("sqsum8U", &sqsum8u)?;
    highgui::imshow("tilted8U", &tilted8u)?;

    highgui::wait_key(0)?;

    Ok(())
}


// 为 Mat 添加转换为 8U 格式的辅助trait
trait MatConversion {
    fn convert_to_8u(&self) -> opencv::Result<Mat>;
}

impl MatConversion for Mat {
    fn convert_to_8u(&self) -> opencv::Result<Mat> {
        let mut result = Mat::default();
        self.convert_to(&mut result, opencv::core::CV_8U, 1.0, 0.0)?;
        Ok(result)
    }
}
