use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Rect, Vec2i, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter8/";

pub(crate) fn run() -> Result<()> {
    // 创建示例矩阵并进行 DFT 变换
    let a = Mat::from_slice_2d(&[
        &[1.0f32, 2.0, 3.0, 4.0, 5.0],
        &[2.0, 3.0, 4.0, 5.0, 6.0],
        &[3.0, 4.0, 5.0, 6.0, 7.0],
        &[4.0, 5.0, 6.0, 7.0, 8.0],
        &[5.0, 6.0, 7.0, 8.0, 9.0]
    ])?;

    let mut b = Mat::default();
    let mut c = Mat::default();
    let mut d = Mat::default();

    // 正变换
    opencv::core::dft(&a, &mut b, opencv::core::DFT_COMPLEX_OUTPUT, 0)?;
    // 逆变换只输出实数
    opencv::core::dft(&b, &mut c, opencv::core::DFT_INVERSE | opencv::core::DFT_SCALE | opencv::core::DFT_REAL_OUTPUT, 0)?;
    // 逆变换
    opencv:: core::idft(&b, &mut d, opencv::core::DFT_SCALE, 0)?;

    // 读取并处理图像
    let img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 转换为灰度图并调整大小
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    imgproc::resize(&gray.clone(), &mut gray, Size::new(502, 502), 0.0, 0.0, imgproc::INTER_LINEAR)?;
    highgui::imshow("原图像", &gray)?;

    // 计算最优 DFT 尺寸, 计算合适的离散傅里叶变换尺寸
    let rows = opencv::core::get_optimal_dft_size(gray.rows())?;
    let cols = opencv::core::get_optimal_dft_size(gray.cols())?;

    // 扩展图像边界
    let mut appropriate = Mat::default();
    let t = (rows - gray.rows()) / 2;  // 上方扩展行数
    let b = rows - gray.rows() - t;    // 下方扩展行数
    let l = (cols - gray.cols()) / 2;  // 左侧扩展行数
    let r = cols - gray.cols() - l;    // 右侧扩展行数

    opencv::core::copy_make_border(
        &gray,
        &mut appropriate,
        t,
        b,
        l,
        r,
        opencv::core::BORDER_CONSTANT,
        Scalar::default()
    )?;
    highgui::imshow("扩展后的图像", &appropriate)?;

    // 准备 DFT 输入
    let mut flo = Vector::<Mat>::new();
    let mut appropriate_float = Mat::default();
    appropriate.convert_to(&mut appropriate_float, opencv::core::CV_32F, 1.0, 0.0)?;
    flo.push(appropriate_float);
    flo.push(Mat::zeros(appropriate.rows(), appropriate.cols(), opencv::core::CV_32F)?.to_mat()?);

    //合成一个多通道矩阵
    let mut complex = Mat::default();
    opencv::core::merge(&flo, &mut complex)?;

    // 执行 DFT, //进行离散傅里叶变换
    let mut result = Mat::default();
    opencv::core::dft(&complex, &mut result, opencv::core::DFT_COMPLEX_OUTPUT, 0)?;

    // 分离实部和虚部
    let mut result_channels = Vector::<Mat>::new();
    opencv::core::split(&result, &mut result_channels)?;

    // 计算幅值
    let mut amplitude = Mat::default();
    opencv::core::magnitude(&result_channels.get(0)?, &result_channels.get(1)?, &mut amplitude)?;

    // 进行对数放缩公式为： M1 = log（1+M），保证所有数都大于0
    opencv::core::add(&amplitude.clone(),
                      &Scalar::new(1.0, 0.0, 0.0, 0.0),
                      &mut amplitude,
                      &opencv::core::no_array(),
                      -1)?;
    opencv::core::log(&amplitude.clone(), &mut amplitude)?;

    // 裁剪到原始尺寸, //与原图像尺寸对应的区域
    let roi = Rect::new(t, l, gray.cols(), gray.rows());
    amplitude = amplitude.roi(roi)?.try_clone()?;

    // 归一化
    opencv::core::normalize(&amplitude.clone(), &mut amplitude, 0.0, 1.0, opencv::core::NORM_MINMAX, -1, &Mat::default())?;
    highgui::imshow("傅里叶变换结果幅值图像", &amplitude)?;

    // 重排象限
    let center_x = amplitude.cols() / 2;
    let center_y = amplitude.rows() / 2;

    // 定义四个象限
    let mut qlt = amplitude.roi(Rect::new(0, 0, center_x, center_y))?.try_clone()?;
    let mut qrt = amplitude.roi(Rect::new(center_x, 0, center_x, center_y))?.try_clone()?;
    let mut qlb = amplitude.roi(Rect::new(0, center_y, center_x, center_y))?.try_clone()?;
    let mut qrb = amplitude.roi(Rect::new(center_x, center_y, center_x, center_y))?.try_clone()?;

    // 交换象限
    let mut med = Mat::default();
    //交换象限，左上和右下进行交换
    qlt.copy_to(&mut med)?;
    qrb.copy_to(&mut qlt)?;
    med.copy_to(&mut qrb)?;
    //交换象限，左下和右上进行交换
    qrt.copy_to(&mut med)?;
    qlb.copy_to(&mut qrt)?;
    med.copy_to(&mut qlb)?;

    highgui::imshow("中心化后的幅值图像", &amplitude)?;

    highgui::wait_key(0)?;

    Ok(())
}