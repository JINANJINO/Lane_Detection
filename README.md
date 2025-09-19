# Lane_Detection
## 1. Perspective Transformation(HomoGraphy)

Definition:
Perspective transformation, also known as planar homography, is a projective mapping that relates the coordinates of points on one plane to the coordinates of points on another plane, as viewed from a camera perspective. Mathematically, it is a linear transformation in homogeneous coordinates that can represent translation, rotation, scaling, and perspective distortion simultaneously.

<img width="382" height="226" alt="image" src="https://github.com/user-attachments/assets/a333e8c4-c297-468f-911e-a7fa2fc970f1" />

**Mathematical Formulation:**
A 2D perspective transformation can be expressed using a **3×3 homography matrix** $$H$$:

$$
H =
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
$$

Where:

- $$(x,y)$$ are the coordinates of a point in the original image plane.
- $$(x′,y')$$ are the coordinates of the corresponding point in the transformed plane.
- $$w′$$ is the homogeneous scale factor. After transformation, actual coordinates are obtained by normalizing: 
$$x′/w′$$ $$y′/w′$$
Since the homography matrix is defined up to scale, it has **8 degrees of freedom** (9 elements minus 1 scale factor). Therefore, at least **4 corresponding points** are required to compute the matrix uniquely.

**Computation:**
To compute a homography matrix $$H$$:

1.Identify at least 4 pairs of corresponding points 

$$
(x_i, y_i) \leftrightarrow (x_i', y_i')
$$


between the source and target planes 

2. Set up a system of linear equations based on the relationship:

$$
x_i' = \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}}, \quad
y_i' = \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}}
$$


3. Solve for the 8 unknowns (since $$h_{33}$$ can be normalized to 1).
4. Use methods like **Direct Linear Transformation (DLT)**, optionally refined by **Levenberg-Marquardt optimization** for better accuracy in the presence of noise

---

## 1-1. Homography Pracice
You can try it out in the ```homograpy.py``` file in my repository.

- **Original Image**
- <img width="1242" height="375" alt="image" src="https://github.com/user-attachments/assets/49f74831-2420-434d-bf88-6783417b391b" />

- **BEV(Bird Eyes View)**
<img width="480" height="480" alt="image" src="https://github.com/user-attachments/assets/d08b90aa-0bbb-4ef4-8303-d25733b72c2d" />

---

## 2. Lane Detection(Hough Transform Line)

<img width="471" height="228" alt="image" src="https://github.com/user-attachments/assets/c5966ca7-e660-490c-bd22-10c7d0f64184" />

**Definition:**
The Hough Transform is a feature extraction technique used in computer vision and image processing to detect **geometric shapes**, most commonly **lines**, within an image. It is particularly effective in identifying lines in **noisy edge-detected images**, where direct slope-intercept methods may fail.

**Basic Idea:**
- A line in Cartesian coordinates can be expressed as:

$$
y = mx + b
$$

where $$m$$ is the slope and $$b$$ is the y-intercept. However, representing vertical lines with infinite slope is problematic.

- Therefore, the Hough Transform uses the polar form of a line:

$$
ρ=xcosθ+ysinθ
$$

where:
- $$ρ$$ is the perpendicular distance from the origin to the line.
- $$θ$$ is the angle between the x-axis and the line's perpendicular.
- Each edge point $$(x,y)$$ in the image votes for all possible $$(ρ,θ)$$ pairs that satisfy the line equation. The collection of votes forms an **accumulator space**, and peaks in this space correspond to the most likely lines.

---

## 2-1.Hough Transform Line Practice

You can follow along by referencing the ```hough_trasnform.py``` file in my repo.

- **Draw Line & Canny Filter**

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/8e2251cf-9a82-4a0c-b89f-99ad6e33da26" /> <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/62ed5c9c-ba80-4366-b918-1c5d42f9b7cd" />

- **Implementation Hough Transfrom & Probabilistic Hough transform**

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/75779117-2f02-4b0d-b0d7-b9e6d252f95b" />

---

## 3. Assignment

Implement lane detection using **BEV transform**, **edge detection**, and **Hough transform**.

- code

```python
############################################################
# Author : jinhanlee
# Date   : 2025-09-19
############################################################

import cv2
import numpy as np

def lane_detection():
    file_name = "./camera.png"
    img = cv2.imread(file_name)
    if img is None:
        print("이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 원본 이미지 출력
    cv2.imshow("Original", img)

    # BEV transform
    src_list = np.float32([
        [488, 192], [580, 192], [142, 373], [820, 373]
    ])
    dst_list = np.float32([
        [0, 0], [480, 0], [0, 480], [480, 480]
    ])
    M = cv2.getPerspectiveTransform(src_list, dst_list)
    bev = cv2.warpPerspective(img, M, (480, 480), flags=cv2.INTER_LINEAR)
    cv2.imshow("BEV", bev)

    # ================= Canny Edge Detection =================
    gray_bev = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    
    blurred_bev = cv2.GaussianBlur(gray_bev, (5, 5), 0)
    edges = cv2.Canny(blurred_bev, 50, 150)
    cv2.imshow('Edges (BEV)', edges)

    # ================= Hough Transform =================
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)  
    edges_hough = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) 
    if lines is not None:
        for rho, theta in lines[:,0]:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(edges_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=20, maxLineGap=60)
    if linesP is not None:
        for x1, y1, x2, y2 in linesP[:,0]:
            cv2.line(edges_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Detected Lanes', edges_hough)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

lane_detection()
```

- **1. Original Image**
  <img width="1242" height="375" alt="image" src="https://github.com/user-attachments/assets/f82ee735-73e5-4cec-8fe1-d3a6c8f0537f" />


- **2. BEV**
<img width="480" height="480" alt="image" src="https://github.com/user-attachments/assets/e747a2f8-9a55-4606-84bd-4327697cc312" />

- **3. Canny Edge Filter**
<img width="480" height="480" alt="image" src="https://github.com/user-attachments/assets/67d15142-7a71-4248-8ee9-0472e108b3f3" />

- **4.Lane Detection**
<img width="480" height="480" alt="image" src="https://github.com/user-attachments/assets/11a2f4bd-c493-4840-ae52-78708837fccc" />


