use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec4i, RNG, Point2f, Point2d, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    // 创建初始图像
    let mut img = Mat::new_rows_cols_with_default(
        500,
        500,
        opencv::core::CV_8UC3,
        Scalar::all(0.0),
    )?;
    
    // 随机数生成器
    let mut rng = RNG::default()?;
    loop {
        // 生成随机点数量
        let count = rng.uniform(1, 101)? as usize;
        let mut points = Vec::with_capacity(count);
        // 生成随机点
        for _ in 0..count {
            let pt = Point::new(
                rng.uniform(img.cols() / 4, img.cols() * 3 / 4)?,
                rng.uniform(img.rows() / 4, img.rows() * 3 / 4)?,
            );
            points.push(pt);
        }
        // 转换点类型
        let points_f32: Vec<Point2f> = points.iter()
            .map(|p| Point2f::new(p.x as f32, p.y as f32))
            .collect();

        // 寻找包围点集的三角形
        let mut triangle = Vector::<Point2f>::new();
        let area = imgproc::min_enclosing_triangle(&Vector::from_slice(points_f32.as_slice()), &mut triangle)?;
        println!("triangle area: {}", area);

        // 寻找包围点集的圆形
        let mut center = Point2f::default();
        let mut radius = 0.0;
        imgproc::min_enclosing_circle(&Vector::from_slice(points_f32.as_slice()), &mut center, &mut radius)?;

        // 重置图像
        img.set_to(&Scalar::all(0.0), &Mat::default())?;

        let mut img2 = img.clone();
        // 在图像中绘制坐标点
        for point in &points {
            imgproc::circle(
                &mut img,
                *point,
                3,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                -1,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::circle(
                &mut img2,
                *point,
                3,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                -1,
                imgproc::LINE_AA,
                0,
            )?;
        }
        // 绘制三角形
        for i in 0..3 {
            let start = Point::new(triangle.get(i)?.x as i32, triangle.get(i)?.y as i32);
            let end = Point::new(
                triangle.get((i + 1) % 3)?.x as i32,
                triangle.get((i + 1) % 3)?.y as i32,
            );
            imgproc::line(
                &mut img,
                start,
                end,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_AA,
                0,
            )?;
        }
        // 绘制圆形
        imgproc::circle(
            &mut img2,
            Point::new(center.x as i32, center.y as i32),
            radius as i32,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            0,
        )?;

        // 显示结果
        highgui::imshow("triangle", &img)?;
        highgui::imshow("circle", &img2)?;
        // 等待按键
        let key = highgui::wait_key(0)?;

        // 检查退出条件
        if key == 27 || key == 'q' as i32 || key == 'Q' as i32 {
            break;
        }
    }

    Ok(())
}