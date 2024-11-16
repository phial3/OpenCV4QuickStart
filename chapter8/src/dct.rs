use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec2i, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter8/";

pub(crate) fn run() -> Result<()> {
    // 创建 奇数核矩阵 5x5
    // error: (-213:The function/feature is not implemented) Odd-size DCT's are not implemented in function 'apply'
    // (code: StsNotImplemented, -213)
    let kernel_data = [
        1.0f32, 2.0, 3.0, 4.0, 5.0,
        2.0, 3.0, 4.0, 5.0, 6.0,
        3.0, 4.0, 5.0, 6.0, 7.0,
        4.0, 5.0, 6.0, 7.0, 8.0,
        5.0, 6.0, 7.0, 8.0, 9.0
    ];
    // 偶数核矩阵 4x4
    let kernel_data = [
        1.0f32, 2.0, 3.0, 4.0,
        2.0, 3.0, 4.0, 5.0,
        3.0, 4.0, 5.0, 6.0,
        4.0, 5.0, 6.0, 7.0
    ];
    let kernel = unsafe { Mat::new_rows_cols_with_data_unsafe_def(4, 4, opencv::core::CV_32F, kernel_data.as_ptr() as *mut _)? };

    // DCT 和 IDCT 变换
    let mut a = Mat::default();
    let mut b = Mat::default();
    opencv::core::dct(&kernel, &mut a, 0).context("DCT计算失败").unwrap();
    opencv::core::idct(&a, &mut b, 0).context("IDCT计算失败").unwrap();

    // 读取图像
    let img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("读入图像出错,请确认图像名称是否正确");
    }
    highgui::imshow("原图像", &img)?;

    // 计算最佳变换尺寸
    let width = 2 * opencv::core::get_optimal_dft_size((img.cols() + 1) / 2)?;
    let height = 2 * opencv::core::get_optimal_dft_size((img.rows() + 1) / 2)?;

    // 扩展图像尺寸
    let t = 0;
    let b = height - t - img.rows();
    let l = 0;
    let r = width - l - img.rows();
    let mut appropriate = Mat::default();
    opencv::core::copy_make_border(
        &img,
        &mut appropriate,
        t,
        b,
        l,
        r,
        opencv::core::BORDER_CONSTANT,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
    )?;

    // 分离通道
    let mut channels = Vector::<Mat>::new();
    opencv::core::split(&appropriate, &mut channels)?;

    // 提取BGR颜色各个通道的值（注意OpenCV是BGR顺序）
    let one = channels.get(0)?;
    let two = channels.get(1)?;
    let three = channels.get(2)?;

    // 进行DCT变换
    let mut one_dct = Mat::default();
    let mut two_dct = Mat::default();
    let mut three_dct = Mat::default();

    // 使用Mat_<float>的等效方式
    let mut one_float = Mat::default();
    let mut two_float = Mat::default();
    let mut three_float = Mat::default();
    one.convert_to(&mut one_float, opencv::core::CV_32F, 1.0, 0.0)?;
    two.convert_to(&mut two_float, opencv::core::CV_32F, 1.0, 0.0)?;
    three.convert_to(&mut three_float, opencv::core::CV_32F, 1.0, 0.0)?;

    opencv::core::dct(&one_float, &mut one_dct, 0)?;
    opencv::core::dct(&two_float, &mut two_dct, 0)?;
    opencv::core::dct(&three_float, &mut three_dct, 0)?;

    // 重新组成三个通道
    let mut channels_dct = Vector::<Mat>::new();

    // 类似于Mat_<uchar>的转换
    let mut one_uchar = Mat::default();
    let mut two_uchar = Mat::default();
    let mut three_uchar = Mat::default();

    one_dct.convert_to(&mut one_uchar, opencv::core::CV_8U, 1.0, 0.0)?;
    two_dct.convert_to(&mut two_uchar, opencv::core::CV_8U, 1.0, 0.0)?;
    three_dct.convert_to(&mut three_uchar, opencv::core::CV_8U, 1.0, 0.0)?;

    channels_dct.push(one_uchar);
    channels_dct.push(two_uchar);
    channels_dct.push(three_uchar);

    // 输出图像
    let mut result = Mat::default();
    opencv::core::merge(&channels_dct, &mut result)?;

    highgui::imshow("DCT图像", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}