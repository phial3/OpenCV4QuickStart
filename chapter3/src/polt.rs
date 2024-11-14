use anyhow::{Result, Error, Context};
use opencv::{
    core::{Point, Point2f, Rect, Scalar, Size, Size2f, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

pub(crate) fn run() -> Result<()> {
    // 创建黑色图像
    let mut img = Mat::zeros(512, 512, opencv::core::CV_8UC3)?.to_mat()?;

    // 绘制圆形
    imgproc::circle(
        &mut img,
        Point::new(50, 50),
        25,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::circle(
        &mut img,
        Point::new(100, 50),
        20,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        4,
        imgproc::LINE_8,
        0,
    )?;

    // 绘制直线
    imgproc::line(
        &mut img,
        Point::new(100, 100),
        Point::new(200, 100),
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_4,
        0,
    )?;

    // 绘制椭圆
    imgproc::ellipse(
        &mut img,
        Point::new(300, 255),
        Size::new(100, 70),
        0.0,
        0.0,
        100.0,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    imgproc::ellipse(
        &mut img,
        Point::new(150, 100),         // 中心点
        Size::new(30, 20),     // 轴的大小
        0.0,                        // 旋转角度
        0.0,                    // 起始角度
        360.0,                // 结束角度
        Scalar::new(0.0, 0.0, 255.0, 0.0), // 颜色
        2,                     // 线宽
        imgproc::LINE_8,                // 线型
        0,                      // shift
    )?;

    // 用点近似椭圆
    let mut points = Vector::<Point>::new();
    imgproc::ellipse_2_poly(
        Point::new(200, 400),
        Size::new(100, 70),
        0,
        0,
        360,
        2,
        &mut points,
    )?;

    // 用直线连接点画出椭圆
    for i in 0..points.len() - 1 {
        if i == points.len() - 1 {
            imgproc::line(
                &mut img,
                points.get(0)?,
                points.get(i)?,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
            break;
        }
        imgproc::line(
            &mut img,
            points.get(i)?,
            points.get(i + 1)?,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
    }

    // 绘制矩形
    imgproc::rectangle(
        &mut img,
        Rect::new(50, 400, 50, 50),
        Scalar::new(125.0, 125.0, 125.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::rectangle(
        &mut img,
        Rect::new(400, 450, 60, 50),
        Scalar::new(0.0, 125.0, 125.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;

    // 绘制多边形
    let mut polygons = Vector::<Vector<Point>>::new();

    // 第一个多边形
    let poly1 = Vector::from_iter(vec![
        Point::new(72, 200),
        Point::new(142, 204),
        Point::new(226, 263),
        Point::new(172, 310),
        Point::new(117, 319),
        Point::new(15, 260),
    ]);

    // 第二个多边形
    let poly2 = Vector::from_iter(vec![
        Point::new(359, 339),
        Point::new(447, 351),
        Point::new(504, 349),
        Point::new(484, 433),
        Point::new(418, 449),
        Point::new(354, 402),
    ]);

    // 第三个多边形
    let poly3 = Vector::from_iter(vec![
        Point::new(350, 83),
        Point::new(463, 90),
        Point::new(500, 171),
        Point::new(421, 194),
        Point::new(338, 141),
    ]);

    polygons.push(poly1);
    polygons.push(poly2);
    polygons.push(poly3);

    imgproc::fill_poly(
        &mut img,
        &polygons,
        Scalar::new(125.0, 125.0, 125.0, 0.0),
        imgproc::LINE_8,
        0,
        Point::new(0, 0),
    )?;

    // 添加文字
    imgproc::put_text(
        &mut img,
        "Learn OpenCV 4",
        Point::new(100, 400),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.0,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    // 显示图像
    highgui::imshow("", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}