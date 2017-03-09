# **Finding Lane Lines on the Road** 

## Xinlin's report

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

# Reflection

## 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

1. Convert the images from **RGB** space to **HSV** space, and use **V channel** instead of **grayscale** image.

  ```python
  # Convert image color space to HSV space
  HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

  H = HSV[:,:,0]
  S = HSV[:,:,1]
  V = HSV[:,:,2]
  ```

  * Reason:
  In my initial solution, I use grayscale image. However, it works very bad in the challenge part. I found that the grayscale image can not provide enough gradient information to distinguish the lane from background in this scenario. Especially after gaussian blur. So I decided to switch to HSV space.

  * The intensity of the grayscale image and the V channel image were plotted along these three lines to show the difference.
  ![Original](./report/Ori111.jpg)
  * The following image shows that the grayscale cannot provide enough gradient information to detect the lane in this scenario.
  ![Grayscale](./report/G111.jpg)
  * The V channel of the image provide better gradient information to distinguish the lane from the background compared with the grayscale image.
  ![V space](./report/V111.jpg)

2. Gaussian blur the V channel image (**kernel size = 3**)
  ```python
  # Gaussian blur
  gaussian = gaussian_blur(V, gaussian_kernel_size)
  ```
  * Reason:
  Detecting the solid lanes are easier than detecting the broken lanes, thus my main focus is finding the threshold to distinguish the broken lane markings from the background. Gaussian blur is used to eliminate the high-frequency noise in the image, however, large Gaussian kernel size no only remove the noise but also blur the broken lane markings. 
  
  * I tested different Gaussian kernel size and picked the kernel size 3. It eliminate part of those unwanted high-frequency noise on the road while preserve the clearness of the broken lane markings
  ![Gaussian](./report/gaussian-comp.jpg)

3. Use Canny edge detector to extract the edges from the image (**low threshold = 90, high threshold = 150**)
  ```python
  # Canny edge detection
  edges = canny(gaussian, canny_low_threshold, canny_high_threshold)
  ```
  * Reason: To determine the proper thresholds of the Canny edge detector, I draw all edges with different gradient thresholds. 
  
  * In the following examples, red edges (50 - 100), purple edges (100 - 150), white edges (> 150)
  ![Canny1](./report/canny1.jpg)
  ![Canny3](./report/canny3.jpg)
  
  * In the Canny edge detector, the edges satisfying high threshold are used as the seed to link those edges satisfying low threshold. Thus, the high threshold should be set to a relatively high value to eliminate noise while low enough to preserve capability that generate the edge seeds on the lane. The low threshold should be set to a high value to eliminate noise while still capable to capture the complete lane (Like the following scenario).
  ![Canny2](./report/canny2.jpg)
  
4. Use a Region of Interest mask to remove non-relevent area 

  ```python
  # Define a convex polygon mask for the Region of Interest
  imshape = image.shape
    
  left_bottom_point = [np.round(0.05 * imshape[1]).astype(int), np.round(0.95 * imshape[0]).astype(int)]
  left_top_point = [np.round(0.46*imshape[1]).astype(int), np.round((0.63)*imshape[0]).astype(int)]
  right_top_point = [np.round(0.54*imshape[1]).astype(int), np.round((0.63)*imshape[0]).astype(int)]
  right_bottom_point = [np.round(0.95 * imshape[1]).astype(int), np.round(0.95 * imshape[0]).astype(int)]
    
  horizontal_shift = np.round(0.02 * imshape[1]).astype(int)
  left_bottom_point[0] += horizontal_shift
  left_top_point[0] += horizontal_shift
  right_top_point[0] += horizontal_shift
  right_bottom_point[0] += horizontal_shift
    
  vertical_shift = -1 * np.round(0.025 * imshape[0]).astype(int)
  left_bottom_point[1] += vertical_shift
  left_top_point[1] += vertical_shift
  right_top_point[1] += vertical_shift
  right_bottom_point[1] += vertical_shift    
    
  #Convert to tuple
  left_bottom_point = tuple(left_bottom_point)
  left_top_point = tuple(left_top_point)
  right_top_point = tuple(right_top_point)
  right_bottom_point = tuple(right_bottom_point)
    
  RoI_vertices = np.array([[left_bottom_point, left_top_point, right_top_point, right_bottom_point]], dtype=np.int32)
  ```

  ```python
  # Region of Interest
  RoI = region_of_interest(edges, RoI_vertices)
  ```
  * Reason: The geometry and position of the region of interest is highly dependent on the position of the camera.
  ![RoI](./report/RoI.jpg)
  
5. Hough Line Transform, **low threshold = 90, high threshold = 150**
  ```python
  # Detect lines by using probabilistic Hough Line Transform   
  lines = hough_lines(RoI, hough_rho, hough_theta, hough_threshold, hough_min_line_len, hough_max_line_gap)
  ```
  * Reason:

 
  
  

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...