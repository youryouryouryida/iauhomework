#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:20:41 2019

@author: l
"""
import cv2
for each in kp_record:
    cv2.circle(img,(each[0],each[1]),int(each[2]),(0,0,100),1)
cv2.imshow('img',img)
cv2.waitKey(0)