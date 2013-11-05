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
  return alpha

def computeConjugateDirectionStepSize(old_r, new_r):
  return beta

def deconvCG(im_blur, kernel, niter=10):
  ''' return deblurred image '''
  return im

def laplacianKernel():
  ''' a 3-by-3 array '''
  return laplacian_kernel

def applyLaplacian(im):
  ''' return Lx (x is im)'''
  return out

def applyAMatrix(im, kernel):
  ''' return Ax, where A = M^TM'''
  return out

def applyRegularizedOperator(im, kernel, lamb):
  ''' (A + lambda L )x'''
  return out


def computeGradientStepSize_reg(grad, p, kernel, lamb):
  return alpha

def deconvCG_reg(im_blur, kernel, lamb=0.05, niter=10):
  ''' return deblurred and regularized im '''

  return im

    
def naiveComposite(bg, fg, mask, y, x):
  ''' naive composition'''
  return out

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



