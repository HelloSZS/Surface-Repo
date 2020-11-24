import os
from selenium import webdriver
import time

driver = webdriver.Chrome()
driver.get("http://xgfy.sit.edu.cn/h5/#/")

time.sleep(2)

search1 = driver.find_element_by_xpath("//input[1]")
search1.send_keys("206141119")
time.sleep(1)
search2 = driver.find_element_by_xpath("//input[@type='password']")
search2.send_keys("a13698472")

time.sleep(2)

btn = driver.find_element_by_xpath("//uni-button[@class='cu-btn lg bg-green shadow']")
btn.click()
time.sleep(2)

btn2 = driver.find_element_by_xpath("//uni-button[@class='cu-btn lg bg-green shadow']")
btn2.click()

time.sleep(4)
btn3 = driver.find_element_by_xpath("//uni-image[@style='width: 60px; height: 60px;']")
btn3.click()


#btn4点击选择位置按钮
time.sleep(2)
btn4 = driver.find_element_by_xpath("//uni-button[@class='cu-btn round bg-green shadow']")
btn4.click()

#选择位置上海市
time.sleep(1)
#uni-view/uni-view[@class='simple-address cu-bar bg-white solid-bottom foot']/uni-view[@class='simple-address-content simple-address--fixed bottom simple-bottom-content content-ani']/uni-view[class='simple-address__box']/uni-picker-view[@class='simple-address-view']/div[@class='uni-picker-view-wrapper']/uni-picker-view-column/div[@class='uni-picker-view-group']/div[@style='padding: 84.5px 0px; transform: translateY(0px) translateZ(0px);']
btn5 = driver.find_element_by_xpath("//uni-picker-view-column/div[@class='uni-picker-view-group']/div[@style='padding: 84.5px 0px; transform: translateY(0px) translateZ(0px);']/uni-view[@class=‘picker-item’ and text()='上海市']")
btn5.click()
h5 = driver.find_element_by_xpath("//uni-view[@class='content']/uni-text/span")
h5.send_keys("上海市-市辖区")

#按下确定
# time.sleep(1)
# btn6 = driver.find_element_by_xpath("//uni-view[@class='simple-address cu-bar bg-white solid-bottom  foot']"
#                                     "/uni-view[@class='simple-address-content simple-address--fixed bottom simple-bottom-content content-ani']"
#                                     "/uni-picker-view[@class='simple-address__header']"
#                                     "/uni-picker-view[@class='simple-address__header-btn-box']"
#                                     "/div[@class='simple-address__header-text']")
# btn6.click()
#
time.sleep(2)
btn7 = driver.find_element_by_xpath("//uni-button[text()='提交']")
# btn7.click()