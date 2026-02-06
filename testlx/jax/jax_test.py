import cv2  # 先加载图像库
import pygame 
import jax
import jax.numpy as jnp
print("--- Step 1: Imports OK ---")
x = jnp.ones((10, 10))
print("--- Step 2: JAX Array OK ---")
cap = cv2.VideoCapture(1) # 尝试开启摄像头（模拟环境初始化）
print("--- Step 3: Camera OK ---")