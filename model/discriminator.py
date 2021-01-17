import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class FCDiscriminator(nn.Module):
    
    def __init__(self, num_classes, ndf = 32):
        super(FCDiscriminator, self).__init__()
        self.num_classes = num_classes
        
        self.conv_list = nn.ModuleList()
        for i in range(num_classes):
            self.conv_list.append(nn.Conv2d(num_classes, ndf, kernel_size=(2,4), stride=2, padding=(0,1)))
            # self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=(2,4), stride=2, padding=(0,1))
            # self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=(2,4), stride=2, padding=1)
            # self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=(2,4), stride=2, padding=1)
            self.conv_list.append(nn.Conv2d(ndf, 1, kernel_size=(2,4), stride=2, padding=(0,1)))
            
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()
        
    @staticmethod #This type of method takes neither a self nor a cls parameter
    def return_objs_contour(pred,num_classes):
        
        pred_np = pred.squeeze(0).permute(1,2,0).cpu().numpy()
        argmax_out = np.asarray(np.argmax(pred_np, axis=2), dtype=np.uint8)
        
        ret_contours = []
        
        for i in range(0,num_classes):
            class_img = argmax_out.copy()
            class_img[np.where(class_img!=i)] = 0
            class_img[np.where(class_img!=0)] = 255
            
            contours = cv2.findContours(class_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            class_contur = []
            for cntr in contours:
                x,y,w,h = cv2.boundingRect(cntr)
                class_contur.append([x,y,w,h])
                # cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # print("x,y,w,h:",x,y,w,h)
                
            ret_contours.append(class_contur)
                
        return ret_contours


    def forward(self, x_):
    
        disc_out = []
        with torch.no_grad():
            contour_list = self.return_objs_contour(x_, self.num_classes)
                    
        for i_class in range(self.num_classes):
            for i_contour in range(len(contour_list[i_class])):
                x= contour_list[i_class][i_contour][0]
                y= contour_list[i_class][i_contour][1]
                w= contour_list[i_class][i_contour][2]
                h= contour_list[i_class][i_contour][3]
                
                if(h>=4 and w>=8):
                    x_out = self.conv_list[2*i_class](x_[:,:,y:y+h+1,x:x+w+1])
                    x_out = self.leaky_relu(x_out)
                    # x = self.conv2(x)
                    # x = self.leaky_relu(x)
                    # x = self.conv3(x)
                    # x = self.leaky_relu(x)
                    # x = self.conv4(x)
                    # x = self.leaky_relu(x)
                    x_out = self.conv_list[(2*i_class)+1](x_out)
                    #x = self.up_sample(x)
                    #x = self.sigmoid(x) 
                    disc_out.append(x_out)
        return disc_out