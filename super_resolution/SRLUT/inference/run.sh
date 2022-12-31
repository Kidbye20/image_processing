g++ -fPIC -shared -O2 fast_lut_interpolation.cpp -o fast_lut_interpolation.so
python inference.py
# python inference_cupy.py