import numpy as np

# === Deconvolution with gradient descent ===

def dotIm(im1, im2):
  return np.sum(im1 * im2)

def applyKernel(im, kernel):
  ''' return Mx, where x is im'''
  return convolve3(im, kernel)

def applyConjugatedKernel(im, kernel):
  ''' return M^T x, where x is im '''
  return convolve3(im, kernel.T)

def computeResidual(kernel, x, y):
  ''' return y - Mx '''
  return y - applyKernel(x, kernel)

def computeStepSize(r, kernel):
  return dotIm(r, r) / dotIm(r, applyConjugatedKernel(applyKernel(r, kernel), kernel))

def deconvGradDescent(im_blur, kernel, niter=1000):
  ''' return deblurred image '''
  M = kernel
  x = np.zeros(np.shape(im_blur))
  y = im_blur
  for step in xrange(niter):
    r = applyConjugatedKernel(computeResidual(M, x, y), M)
    alpha = computeStepSize(r, M)
    x += alpha * r
  return x

# === Deconvolution with conjugate gradient ===

def computeGradientStepSize(r, d, kernel):  
  return dotIm(r, r) / dotIm(d, applyConjugatedKernel(applyKernel(d, kernel), kernel))

def computeConjugateDirectionStepSize(old_r, new_r):
  return dotIm(new_r, new_r) / dotIm(old_r, old_r)

def deconvCG(im_blur, kernel, niter=10):
  ''' return deblurred image '''
  #Adi = applyConjugatedKernel(applyKernel(di, kernel), kernel)
  M = kernel
  x = np.zeros(np.shape(im_blur))
  y = im_blur
  d = applyConjugatedKernel(y, M) - applyConjugatedKernel(applyKernel(x, M), M)
  r_old = d.copy()
  r_new = d.copy()
  for step in xrange(niter):
    alpha = computeGradientStepSize(r_new, d, M)
    beta = computeConjugateDirectionStepSize(r_old, r_new)
    x += alpha * d
    r_old = r_new.copy()
    r_new = r_old - alpha * applyConjugatedKernel(applyKernel(d, M), M)
    d = r_new + beta * d    
  return x

def laplacianKernel():
  ''' a 3-by-3 array '''
  return np.array([[ 0, -1,  0],
                   [-1,  4, -1],
                   [ 0, -1,  0]]) 

def applyLaplacian(im):
  ''' return Lx (x is im)'''
  return applyKernel(im, laplacianKernel()) 

def applyAMatrix(im, kernel):
  ''' return Ax, where A = M^TM'''
  return applyConjugatedKernel(applyKernel(im, kernel), kernel)

def applyRegularizedOperator(im, kernel, lamb):
  ''' (A + lambda L )x'''
  return applyAMatrix(im, kernel) + lamb * applyLaplacian(im)


def computeGradientStepSize_reg(grad, p, kernel, lamb):
  ''' returns alpha '''
  return dotIm(grad, grad) / dotIm(p, applyRegularizedOperator(p, kernel, lamb))

def deconvCG_reg(im_blur, kernel, lamb=0.05, niter=10):
  ''' return deblurred and regularized im '''
  M = kernel
  x = np.zeros(np.shape(im_blur))
  y = im_blur
  d = applyRegularizedOperator(y, M, lamb) - applyAMatrix(x, M)
  r_old = d.copy()
  r_new = d.copy()
  for step in xrange(niter):
    alpha = computeGradientStepSize_reg(r_new, d, M, lamb)    
    beta = computeConjugateDirectionStepSize(r_old, r_new)
    x += alpha * d
    r_old = r_new.copy()
    r_new = r_old - alpha * applyAMatrix(d, M)
    d = r_new + beta * d    
  return x

    
def naiveComposite(bg, fg, mask, y, x):
  ''' naive composition'''    
  (height, width, rgb) = np.shape(fg)  
  validPixels = fg * mask
  invalidPixels = bg[y:y+height, x:x+width] * (1 - mask)  
  bg[y:y+height, x:x+width] = validPixels
  bg[y:y+height, x:x+width] += invalidPixels
  return bg

def Poisson(bg, fg, mask, niter=200):
  ''' Poisson editing using gradient descent'''
            
  return x



def PoissonCG(bg, fg, mask, niter=200):
  ''' Poison editing using conjugate gradient '''

  return x 
  
  

#==== Helpers. Use them as possible. ==== 

def convolve3(im, kernel):
  from scipy import ndimage
  center=(0,0)
  r=ndimage.filters.convolve(im[:,:,0], kernel, mode='reflect', origin=center) 
  g=ndimage.filters.convolve(im[:,:,1], kernel, mode='reflect', origin=center) 
  b=ndimage.filters.convolve(im[:,:,2], kernel, mode='reflect', origin=center) 
  return (np.dstack([r,g,b]))

def gauss2D(sigma=2, truncate=3):
  kernel=horiGaussKernel(sigma, truncate);
  kerker=np.dot(kernel.transpose(), kernel)
  return kerker/sum(kerker.flatten())

def horiGaussKernel(sigma, truncate=3):
  from scipy import signal
  sig=signal.gaussian(2*int(sigma*truncate)+1,sigma)
  return np.array([sig/sum(sig)])



