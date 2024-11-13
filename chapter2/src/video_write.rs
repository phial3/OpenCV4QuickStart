use anyhow::Result;
use opencv::{prelude::*, videoio, highgui};


pub(crate) fn run() -> Result<()> {
    let window = "video capture";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    // 0 is the default camera
    let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&capture)? {
        panic!("Unable to open default camera!");
    }


    // 获取第一帧图像，检查视频流
    let mut img = Mat::default();
    capture.read(&mut img)?;

    if img.empty() {
        println!("没有获取到图像");
        return Ok(());
    }

    // 判断相机（视频）是否为彩色图像
    let is_color = img.typ() == opencv::core::CV_8UC3;
    // 设置视频保存配置, MJPG 编码格式
    let codec = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?;
    let fps = 25.0;
    let filename = "live.avi"; // 保存的视频文件名称

    // 创建视频写入对象
    let mut writer = videoio::VideoWriter::new(filename, codec, fps, img.size()?, is_color)?;
    // 判断视频流是否创建成功
    if !writer.is_opened()? {
        println!("打开视频文件失败，请确实是否为合法输入");
        return Ok(());
    }

    // 循环读取视频流并保存
    loop {
        if !capture.read(&mut img)? {
            println!("摄像头断开连接或者视频读取完成");
            break;
        }

        writer.write(&img)?; // 写入视频流

        // 显示图像
        highgui::imshow("Live", &img)?;

        // 按ESC键退出
        let key = highgui::wait_key(50)?;
        if key == 27 {  // ESC 键的 ASCII 码为27
            break;
        }

        // 加点延时，避免 CPU 占用过高
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    // 退出时自动释放摄像头和视频流
    writer.release()?;
    capture.release()?;

    Ok(())
}