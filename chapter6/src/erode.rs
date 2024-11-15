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
    // 生成用于腐蚀的原图像
    let mut src = Mat::zeros_size(Size::new(6, 6), opencv::core::CV_8U)?.to_mat()?;
    {
        let data = vec![
            0, 0, 0, 0, 255, 0,
            0, 255, 255, 255, 255, 255,
            0, 255, 255, 255, 255, 0,
            0, 255, 255, 255, 255, 0,
            0, 255, 255, 255, 255, 0,
            0, 0, 0, 0, 0, 0
        ];
        src.data_typed_mut::<u8>()?.copy_from_slice(&data);
    }
    // 创建结构元素
    let struct1 = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(3, 3),
        Point::new(-1, -1)
    )?;
    let struct2 = imgproc::get_structuring_element(
        imgproc::MORPH_CROSS,
        Size::new(3, 3),
        Point::new(-1, -1)
    )?;
    // 腐蚀
    let mut erode_src = Mat::default();
    imgproc::erode(&src, &mut erode_src, &struct2, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    // 显示原图和腐蚀后的图像
    highgui::named_window("src", highgui::WINDOW_GUI_NORMAL)?;
    highgui::named_window("erodeSrc", highgui::WINDOW_GUI_NORMAL)?;
    highgui::imshow("src", &src)?;
    highgui::imshow("erodeSrc", &erode_src)?;

    // 读取图像
    let learn_cv_black = imgcodecs::imread(&format!("{}{}", BASE_PATH, "LearnCV_black.png"), imgcodecs::IMREAD_ANYCOLOR)?;
    let learn_cv_white = imgcodecs::imread(&format!("{}{}", BASE_PATH, "LearnCV_white.png"), imgcodecs::IMREAD_ANYCOLOR)?;
    let mut img = imgcodecs::imread(&format!("{}{}", BASE_PATH, "rice.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() || learn_cv_black.empty() || learn_cv_white.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 黑背景图像腐蚀
    let mut erode_black1 = Mat::default();
    let mut erode_black2 = Mat::default();
    imgproc::erode(&learn_cv_black, &mut erode_black1, &struct1, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    imgproc::erode(&learn_cv_black, &mut erode_black2, &struct2, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::imshow("LearnCV_black", &learn_cv_black)?;
    highgui::imshow("erode_black1", &erode_black1)?;
    highgui::imshow("erode_black2", &erode_black2)?;

    // 白背景腐蚀
    let mut erode_write1 = Mat::default();
    let mut erode_write2 = Mat::default();
    imgproc::erode(&learn_cv_white, &mut erode_write1, &struct1, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    imgproc::erode(&learn_cv_white, &mut erode_write2, &struct2, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::imshow("LearnCV_write", &learn_cv_white)?;
    highgui::imshow("erode_write1", &erode_write1)?;
    highgui::imshow("erode_write2", &erode_write2)?;

    let mut img2 = img.clone();  // 克隆一个单独的图像，用于后期图像绘制
    let mut rice = Mat::default();
    let mut rice_bw = Mat::default();

    // 将图像转成二值图像，用于统计连通域
    imgproc::cvt_color(&img, &mut rice, imgproc::COLOR_BGR2GRAY, 0)?;
    imgproc::threshold(&rice, &mut rice_bw, 50.0, 255.0, imgproc::THRESH_BINARY)?;
    let mut out = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();

    // 统计图像中连通域的个数
    let number = imgproc::connected_components_with_stats(&rice_bw, &mut out, &mut stats, &mut centroids, 8, opencv::core::CV_16U)?;
    draw_state(&mut img2, number, &centroids, &stats, "未腐蚀时统计连通域")?;  // 绘制图像

    // 对图像进行腐蚀
    imgproc::erode(&rice_bw.clone(), &mut rice_bw, &struct1, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    let number_after = imgproc::connected_components_with_stats(&rice_bw, &mut out, &mut stats, &mut centroids, 8, opencv::core::CV_16U)?;
    draw_state(&mut img2, number_after, &centroids, &stats, "腐蚀后统计连通域")?;  // 绘制图像

    highgui::wait_key(0)?;

    Ok(())
}

// 绘制包含区域函数
fn draw_state(img: &mut Mat, number: i32, centroids: &Mat, stats: &Mat, str_title: &str) -> Result<()> {
    let mut rng = RNG::new(10086)?;

    // 生成随机颜色
    let mut colors = Vec::with_capacity(number as usize);
    for _ in 0..number {
        let vec3 = Vec3b::from_array([
            rng.uniform(0, 256)? as u8,
            rng.uniform(0, 256)? as u8,
            rng.uniform(0, 256)? as u8,
        ]);
        colors.push(vec3);
    }
    for i in 1..number {
        // 中心位置
        let center_x = centroids.at_2d::<f64>(i, 0)?;
        let center_y = centroids.at_2d::<f64>(i, 1)?;
        // 矩形边框
        let x = stats.at_2d::<i32>(i, 0)?;
        let y = stats.at_2d::<i32>(i, 1)?;
        let w = stats.at_2d::<i32>(i, 2)?;
        let h = stats.at_2d::<i32>(i, 3)?;

        // 中心位置绘制
        imgproc::circle(
            img,
            Point::new(*center_x as i32, *center_y as i32),
            2,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            8,
            0
        )?;

        // 外接矩形
        let rect = Rect::new(*x, *y, *w, *h);
        imgproc::rectangle(
            img,
            rect,
            Scalar::new(
                colors[i as usize][0] as f64,
                colors[i as usize][1] as f64,
                colors[i as usize][2] as f64,
                0.0
            ),
            1,
            8,
            0
        )?;

        // 绘制文本
        imgproc::put_text(
            img,
            &format!("{}", i),
            Point::new(*center_x as i32, *center_y as i32),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            1,
            8,
            false
        )?;
    }

    highgui::imshow(str_title, img)?;

    Ok(())
}