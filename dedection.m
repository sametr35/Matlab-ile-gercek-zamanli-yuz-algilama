web=webcam();

dete =vision.CascadeObjectDetector("face");


while true
        
    im =snapshot(web);
    grayim = rgb2gray(im);
    bbox = step(dete,grayim);
   
        
    
    G = imresize(im, [224, 224]);
        
      
    detpic=insertObjectAnnotation(im,"rectangle",bbox,"Nose");
    imshow(detpic);
end
    
        


