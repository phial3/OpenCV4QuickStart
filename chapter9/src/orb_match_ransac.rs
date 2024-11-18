use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Point2f, Scalar, RNG, DMatch, Ptr, Vector, KeyPoint},
    imgcodecs,
    imgproc,
    highgui,
    calib3d,
    features2d::{self, Feature2D, FlannBasedMatcher, DrawMatchesFlags, ORB, BFMatcher, ORB_ScoreType},
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter9/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img1 = imgcodecs::imread(&format!("{}box.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    let img2 = imgcodecs::imread(&format!("{}box_in_scene.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img1.empty() || img2.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    // 提取ORB特征点
    let mut keypoints1 = Vector::new();
    let mut keypoints2 = Vector::new();
    let mut descriptions1 = Mat::default();
    let mut descriptions2 = Mat::default();

    // 计算特征点
    orb_features(&img1, &mut keypoints1, &mut descriptions1)?;
    orb_features(&img2, &mut keypoints2, &mut descriptions2)?;

    // 特征点匹配
    let mut matches = Vector::new();
    let mut matcher = BFMatcher::create(opencv::core::NORM_HAMMING, false)?;
    matcher.match_(&descriptions1,  &mut matches, &opencv::core::no_array())?;
    matcher.match_(&descriptions2,  &mut matches, &opencv::core::no_array())?;
    println!("matches={}", matches.len());

    // 最小汉明距离筛选
    let good_min = match_min(&matches)?;
    println!("good_min={}", good_min.len());

    // RANSAC算法筛选
    let good_ransac = ransac(&good_min, &keypoints1, &keypoints2)?;
    println!("good_ransac={}", good_ransac.len());

    // 绘制匹配结果
    let mut outimg = Mat::default();
    let mut outimg1 = Mat::default();
    let mut outimg2 = Mat::default();

    features2d::draw_matches(
        &img1, &keypoints1,
        &img2, &keypoints2,
        &matches, &mut outimg,
        Scalar::all(-1.0),
        Scalar::all(-1.0),
        &Vector::new(),
        DrawMatchesFlags::DEFAULT,
    )?;

    features2d::draw_matches(
        &img1, &keypoints1,
        &img2, &keypoints2,
        &good_min, &mut outimg1,
        Scalar::all(-1.0),
        Scalar::all(-1.0),
        &Vector::new(),
        DrawMatchesFlags::DEFAULT,
    )?;

    features2d::draw_matches(
        &img1, &keypoints1,
        &img2, &keypoints2,
        &good_ransac, &mut outimg2,
        Scalar::all(-1.0),
        Scalar::all(-1.0),
        &Vector::new(),
        DrawMatchesFlags::DEFAULT,
    )?;

    highgui::imshow("未筛选结果", &outimg)?;
    highgui::imshow("最小汉明距离筛选", &outimg1)?;
    highgui::imshow("ransac筛选", &outimg2)?;

    highgui::wait_key(0)?;

    Ok(())
}

fn match_min(matches: &Vector<DMatch>) -> Result<Vector<DMatch>> {
    let mut min_dist = f64::MAX;
    let mut max_dist = 0f64;

    // 找出最大和最小距离
    for m in matches.iter() {
        let dist = m.distance as f64;
        min_dist = min_dist.min(dist);
        max_dist = max_dist.max(dist);
    }
    println!("min_dist={}", min_dist);
    println!("max_dist={}", max_dist);

    // 筛选好的匹配点
    let mut good_matches = Vector::new();
    for m in matches.iter() {
        if (m.distance as f64) <= f64::max(2.0 * min_dist, 20.0) {
            good_matches.push(m.clone());
        }
    }

    Ok(good_matches)
}

fn ransac(
    matches: &Vector<DMatch>,
    query_keypoints: &Vector<KeyPoint>,
    train_keypoints: &Vector<KeyPoint>,
) -> Result<Vector<DMatch>> {
    // 提取匹配点对的坐标
    let mut src_points = Vector::<Point2f>::new();
    let mut dst_points = Vector::<Point2f>::new();

    for m in matches.iter() {
        src_points.push(query_keypoints.get(m.query_idx as usize).unwrap().pt());
        dst_points.push(train_keypoints.get(m.train_idx as usize).unwrap().pt());
    }

    println!("query_keypoints={}", src_points.len());
    println!("train_keypoints={}", train_keypoints.len());
    println!("src_points={}", src_points.len());
    println!("dst_points={}", dst_points.len());

    // FIXME: error: (-28:Unknown error code -28) The input arrays should have at least 4 corresponding point sets to calculate Homography in function 'findHomography'
    //      (code: StsVecLengthErr, -28)
    // 使用RANSAC算法计算单应性矩阵，并获取内点掩码
    let mut inliers_mask = Mat::default();
    calib3d::find_homography_ext(
        &Mat::from_slice(src_points.as_slice())?,
        &Mat::from_slice(dst_points.as_slice())?,
        calib3d::RANSAC,
        5.0,
        &mut inliers_mask,
        2000,
        0.995,
    ).context("RANSAC计算单应性矩阵失败").unwrap();

    // 根据掩码筛选匹配点
    let mut matches_ransac = Vector::new();
    for i in 0..matches.len() {
        if *inliers_mask.at_2d::<u8>(i as i32, 0)? != 0 {
            matches_ransac.push(matches.get(i)?);
        }
    }

    Ok(matches_ransac)
}

fn orb_features(gray: &Mat, keypoints: &mut Vector<KeyPoint>, descriptions: &mut Mat) -> Result<()> {
    let mut orb = ORB::create(1000, 1.2f32, 8, 31, 0, 2, ORB_ScoreType::HARRIS_SCORE, 31, 20)?;
    orb.detect(&gray, keypoints, &opencv::core::no_array())?;
    orb.compute(&gray, keypoints, descriptions)?;
    Ok(())
}
