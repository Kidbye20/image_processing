IMAGE AND VISION COMPUTING LABORATORY
at The University of Waterloo


-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2015 The University of Waterloo
All rights reserved.

Permission is hereby granted, without written agreement and without
license or royalty fees, to use, copy, modify, and distribute this
database (the images, the results and the source files) and its 
documentation for any purpose, provided that the copyright 
notice in its entirity appear in all copies of this 
database, and the original source of this database,  
Image and Vision Computing Laboratory (IVC, https://ece.uwaterloo.ca/~z70wang/) at the 
University of Waterloo (UW, http://www.uwaterloo.ca), 
is acknowledged in any publication that reports research using this database.
The database is to be cited in the bibliography as:

Kede Ma, Kai Zeng and Zhou Wang, "Perceptual Quality Assessment for 
Multi-Exposure Image Fusion," IEEE Trans. on Image Processing (TIP), 2015.

Kai Zeng, Kede Ma, Rania Hassen and Zhou Wang, "Perceptual Evaluation of Multi-exposure 
Image Fusion Algorithms," The 6th International Workshop on Quality of Multimedia 
Experience (QoMEX), 2014

IN NO EVENT SHALL THE UNIVERSITY OF WATERLOO BE LIABLE TO ANY PARTY 
FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES 
ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF 
THE UNIVERSITY OF WATERLOO  HAS BEEN ADVISED OF THE POSSIBILITY OF 
SUCH DAMAGE.

THE UNIVERSITY OF WATERLOO SPECIFICALLY DISCLAIMS ANY WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE
PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
WATERLOO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------

Please contact Kede Ma (k29ma@uwaterloo.ca) if you have any questions.
This investigators on this research were:
Dr. Kai Zeng (kzeng@uwaterloo.ca) -- Department of ECE at UW
Dr. Kede Ma (k29ma@uwaterloo.ca) -- Department of ECE at UW
Dr. Rania Hassen (rhassen@uwaterloo.ca) -- Department of ECE at UW
Dr. Zhou Wang (z70wang@uwaterloo.ca) -- Department of ECE at UW

-------------------------------------------------------------------------

17 high-quality source image sequences of maximum size of 384 × 512 × 30 are selected to 
cover diverse image content including natural sceneries, indoor and outdoor 
views, and man-made architectures. 

The origins of the sequences are specified in the above TIP paper. 

8 fusion algorithms are selected, which include simple operators
such as 1) local energy weighted linear combination and
2) global energy weighted linear combination, as well as advanced
MEF algorithms such as 3) Raman09, 4) Gu12, 5) ShutaoLi12, 6) ShutaoLi13 (Li13), 
7) Li12, and 8) Mertens07. These algorithms are chosen to cover a diverse 
types of MEF methods in terms of methodology and behavior. Eventually, a total of 136 
fused images are generated, which are divided into 17 image sets of 8 images each, where 
the images in the same set are created from the same source image sequence. 

The mean opinion scores (MOS) of fused images range from 1 to 10, where 1 
denotes the worst quality and 10 the best. 
------------------------------------------------------------------------------------------------------
source image sequences folder: includes 17 subfolders (in the name of sequenceName_imageOrigin), each 
			       of which contains one source image sequence;
			       
fused images folder	     : contains 136 fused images with their image names well organized in imgName.mat;

MOS.mat                      : contains MOSs of 136 fused images organized according to imgName.mat;

imgName.mat                  : summarizes the image names in the fused images folder.

------------------------------------------------------------------------------------------------

Please refer to the above papers for more details.