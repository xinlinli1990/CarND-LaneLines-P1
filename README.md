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

My pipeline consisted of 7 steps. 

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
  ![Original](./report/Ori111.JPG)
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
  Detecting the solid lanes are easier than detecting the broken lanes, thus my main focus is finding the threshold to distinguish the broken lane markings from the background. Gaussian blur is used to eliminate the high-frequency noise in the image, however, large Gaussian kernel size not only remove the noise but also blur the broken lane markings. 
  
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
  
  * Vertical shift and horizontal shift are added in the *challenge.mp4* test case to guarantee the lanes stay in the center of the region of interest.
  
  ```python
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
  ```
  ![RoI](./report/RoI.JPG)
  
5. Use Hough line transform to detect the line segments from the images, **rho = 2, theta = np.pi/180, threshold = 40, min line len = 10, max line gap = 10**
  ```python
  # Detect lines by using probabilistic Hough Line Transform   
  lines = hough_lines(RoI, hough_rho, hough_theta, hough_threshold, hough_min_line_len, hough_max_line_gap)
  ```
  * Reason: I begin with a set of relatively loose parameters that allow the Hough line transform capture the targeted line segments robustly. Then I observe the difference between the targeted line segments and the noisy line segments then narrow down the range of parameters.

  * The Hough line transform provide good results when testing with the test images.
  ![Hough-Comp](./report/Hough-Comp.jpg)
  
6. Filter background line segments and decide which lane the segments belongs to.
  * Reason: For simple scenarios, the previous 5 steps are good enough to detect the lanes. However, in complex scenarios (like tree shadow or damaged road), lots of errors can be observed. 
  ![Hough-extra](./report/Hough-extra.jpg)
  
  * To filter these unwanted line segments, I set a criteria that remove all semi-vertical and semi-horizontal line segments (angle less than 20 degrees). 
  
  * Then distinguish the line segments belongs to left lane or right lane by positive or negative slope.
  
  ```python
	def filter_and_distribute(lines, img):   
		# Criteria for vertical and horizontal lines
		RAD_TOL = 20 / 180 * np.pi
		TOL = np.sin(RAD_TOL)
		
		img_x_mid = img.shape[1] / 2
		
		left_lane_segments = []
		right_lane_segments = []

		for line in lines:
			for x1,y1,x2,y2 in line:         
				length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
				
				# Eliminate all vertical and horizontal lines
				if (np.abs(x2 - x1) / length < TOL) or (np.abs(y2 - y1) / length < TOL):
					continue               
				
				slope = (y2 - y1) / (x2 - x1)

				x_mid = (x1 + x2) / 2
				
				# Distinguish the line belongs to left lane or right lane
				# Negative slope and on the left side => left lane
				if slope < 0 and x_mid < img_x_mid:
					left_lane_segments.append([[x1, y1, x2, y2]])
					cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
				# Positive slope and on the right side => right lane
				elif slope > 0 and x_mid > img_x_mid:       
					right_lane_segments.append([[x1, y1, x2, y2]])
					cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
				else:
					pass

		return left_lane_segments, right_lane_segments
  ```
  
  * The following image is the result of filter and distribute step. the left lane segments are drawed in green and the right lane segments are drawed in purple.
  ![filter-distribute](./report/filter-and-distribute.jpg)
  
7. Draw the left and right lane segments separately, and use Hough Line Transform again to detect the left and right lanes.
  ```python
  # Detect lines by using probabilistic Hough Line Transform   
  lines = hough_lines(RoI, hough_rho, hough_theta, hough_threshold, hough_min_line_len, hough_max_line_gap)
  ```

  ![filter-distribute-2](./report/filter-and-distribute-2.png)
  * Draw the left and right lane separately. Use large thickness increase the tolerance 
  ![2hough-1](./report/2hough-1.jpg)
  * Use Hough Transform again to extract the lane.
  ![2hough-2](./report/2hough-2.jpg)
  
8. Extrapolate the line segments from second hough transform

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...