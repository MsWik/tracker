import cv2
from math import sqrt
import traceback
import time

class TestTracke():
    def __init__(self):
        self.log_level = 1
        self.tracker_type='CSRT'
        self.tracker=self.get_traccker(self.tracker_type)
        self.trec_on=False
        self.global_x=0
        self.global_y=0
        #self.cap = cv2.VideoCapture('baran.mp4')
        self.cap = cv2.VideoCapture(0)
        ##########################################
        self.frame_rate=int(self.cap.get(5))




        ###########################################
        self.log_patch = "log.txt"
        self.log_file = open(self.log_patch, 'w', encoding="utf-8")
        self.size_G=64
        self.size_A=10
        self.y_c=0
        self.x_c=0
        self.experemental_size=256

        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

        self.all_in_=0


    def get_traccker(self,tracker_type):
        if tracker_type == 'BOOSTING':
            tracker_class = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker_class = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker_class = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker_class = cv2.TrackerTLD_create()
        elif tracker_type == 'GOTURN':
            tracker_class = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker_class = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'MOSSE':
            tracker_class = cv2.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            tracker_class = cv2.TrackerCSRT_create()
        return tracker_class




    def gstreamer_pipeline(self,
            capture_width=1920,
            capture_height=1080,
            display_width=1280,
            display_height=720,
            framerate=30,
            flip_method=2,
    ):
        return (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
        )


    def xywh2xy(self,box):
            x_c=box[0]+box[2]/2
            y_c=box[1]+box[3]/2
            return int(x_c),int(y_c)

    def drawBox(self,img, bbox):
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 0, 255), 2, 2)

            x_black=x-int(self.size_A)
            y_black=y-int(self.size_A)


            cv2.rectangle(img, (x_black, y_black), ((x + w+self.size_A), (y + w+self.size_A)), (0, 0, 0), 1, 1)
            cv2.putText(img, "Tracking", (x_black, y_black-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)




            cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            str_out = 'P(' + str(x) + ',' + str(y) + ')'
            cv2.putText(self.img, str_out, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 0), 1)
            str_bottom='S(' + str(w) + ',' + str(h) + ')'
            cv2.putText(self.img, str_bottom, (x, y+ h+ 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 0), 1)
            #if x>img.shape[1] and y>img.shape[0]:
            #print(x+w,y+h)
            return img

    def nothing(self,*arg):
        pass


    def set_tracker(self,val):
         self.tracker_type = self.tracker_types[val]


    def all_in(self,val):
         self.all_in_ = val




    def set_size_G(self,val):
         self.size_G = max(val, 10)
    def set_size_A(self,val):
         self.size_A = max(val, 10)

    def draw_box(self,event, x, y, flags, param):

        if event == cv2.EVENT_MOUSEMOVE:
            self.global_x = x
            self.global_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox = (self.global_x -int(self.size_G/2), self.global_y - int(self.size_G/2), int(self.size_G), int(self.size_G))
            self.trec_on = True

            ################# EXPERIMENT ####################
            if not self.all_in_:
                    self.tracker.init(self.img, bbox)
            else:
                    crop_img=self.crop_image(self.img)
                    crop_bbox=self.crop_bbox(bbox)
                    print(crop_img.shape)
                    self.tracker.init(crop_img, crop_bbox)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.trec_on = False
            self.tracker = self.get_traccker(self.tracker_type)
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags>0:
                if self.size_G<256:
                        self.size_G+=4
            else:
                if  self.size_G>10:
                        self.size_G-=4



    def crop_image(self,img):
        crop_image=img[self.global_y-int(self.experemental_size/2):self.global_y+int(self.experemental_size/2),  \
                   self.global_x-int(self.experemental_size/2):self.global_x+int(self.experemental_size/2)]
        return crop_image

    def crop_bbox(self,bbox):
        x=int(self.experemental_size/2)
        y=int(self.experemental_size/2)
        w,h=int(bbox[2]), int(bbox[3])
        return (x,y,w,h)


    def crop_image_final(self,img):
        if self.y_c!=0 and self.x_c!=0:
            crop_image = img[self.y_c - int(self.experemental_size / 2):self.y_c + int(
                self.experemental_size / 2),
                         self.x_c - int(self.experemental_size / 2):self.x_c + int(
                             self.experemental_size / 2)]
        else:
            crop_image = img[self.global_y - int(self.experemental_size / 2):self.global_y + int(
                self.experemental_size / 2),
                         self.global_x - int(self.experemental_size / 2):self.global_x + int(
                             self.experemental_size / 2)]
        return crop_image

    def main(self):
        prev=0
        def back(*args):
            pass
        cv2.namedWindow("settings")  # создаем окно настроек
        cv2.createTrackbar('size_G', 'settings', 64, 256, self.set_size_G)
        cv2.createTrackbar('size_A', 'settings', 10, 128, self.set_size_A)
        cv2.createTrackbar('TRACKER_TUPE', 'settings', 7, 7, self.set_tracker)
        cv2.createTrackbar('ALL_IN', 'settings', 0, 1, self.all_in)







        self.log_patch = "log.txt"
        self.log_file = open(self.log_patch, 'a+', encoding="utf-8")

        try:

            success, frame = self.cap.read()

            print(self.cap.isOpened())

            width = int(self.cap.get(3))
            height = int(self.cap.get(4))
            frame_fps = int(self.cap.get(5))
            new_patch_video = 'new_video.avi'
            vw = cv2.VideoWriter(new_patch_video, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), frame_fps,
                                 (width, height))

            window_name = 'Tracking'
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.draw_box)

            while self.cap.isOpened():

                #self.size_G = cv2.getTrackbarPos('size_G', 'settings')





                timer = cv2.getTickCount()

                time_elapsed = time.time() - prev


                if time_elapsed > 1. / self.frame_rate:
                    success, self.img = self.cap.read()
                    prev = time.time()
                    if success:

                        #crop_img=self.crop_image_final(self.img)
                        #succes, bbox = self.tracker.update(crop_img)

                        if not self.all_in_:
                                succes, bbox = self.tracker.update(self.img)

                                if succes:
                                    self.img =self.drawBox(self.img, bbox)
                                    self.x_c,self.y_c=self.xywh2xy(bbox)
                                    self.img[self.y_c,self.x_c]=[255,255,255]
                                else:
                                    cv2.putText(self.img, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                                 crop_img=self.crop_image_final(self.img)
                                 succes, bbox = self.tracker.update(crop_img)
                                 if succes:
                                     self.img = self.drawBox(crop_img, bbox)
                                     #self.x_c, self.y_c = self.xywh2xy(bbox)
                                     #self.img[self.y_c, self.x_c] = [255, 255, 255]
                                 else:
                                     cv2.putText(self.img, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                 (0, 0, 255), 2)



                        cv2.rectangle(self.img, (15, 15), (200, 120), (255, 0, 255), 2)
                        cv2.putText(self.img, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);
                        cv2.putText(self.img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);
                        cv2.putText(self.img, "Tracker:", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);
                        cv2.putText(self.img, self.tracker_type, (120, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2);

                        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
                        if fps > 60:
                            myColor = (20, 230, 20)
                        elif fps > 20:
                            myColor = (230, 20, 20)
                        else:
                            myColor = (20, 20, 230)
                        cv2.putText(self.img, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2);
                        #vw.write(img)

                        if self.global_x != 0 and self.global_y != 0:
                            cv2.rectangle(self.img, (self.global_x - int(self.size_G/2), self.global_y - int(self.size_G/2)), (self.global_x + int(self.size_G/2), self.global_y + int(self.size_G/2)), (255, 255, 255),
                                          1)


                        cv2.imshow(window_name, self.img)
                        if cv2.waitKey(1) & 0xff == ord('q'):
                            break
                        # if 0==cv2.EVENT_MOUSEMOVE:
                        #    print(1111)


                    else:
                        break
                else:
                    pass
        except Exception as e:
            print(traceback.format_exc(), file=self.log_file)
            return {"status": "Error", "message": str(e)}
        finally:
            self.log_file.close()


        return {"status": "OK"}



if __name__ == "__main__":
        test_tracker=TestTracke()
        test_tracker.main()



