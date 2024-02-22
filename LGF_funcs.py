#%matplotlib inline
import mpmath as mp
import math

import numpy as np
import scipy


def eval_lgf(c, alpha, n, m, n_pts = 1e10):
    #theta = np.linspace(-np.pi, np.pi, int(n_pts), endpoint=False, dtype=np.longdouble)
    #print(theta.shape)

    theta = np.linspace(-np.pi, np.pi, int(n_pts), endpoint=False)

    lbd = 2 + 2*alpha
    #theta = np.linspace(-np.pi, np.pi, n_pts + 1, endpoint=True)
    a = alpha*np.cos(theta)*(0-2) + lbd + c**2
    K = (a + np.sqrt(np.square(a) - 4))/2
    I = ( np.cos(theta*n) * (1/K)**m) / (K - 1/K)
    #res = np.trapz(I, theta)
    res = np.sum(I)*(1/len(theta))
    return np.real(res)

def eval_lgf_appell(c, n, m, alpha):
    lbd = 2 + 2*alpha
    c_ratio = lbd/(lbd+c**2)
    a = 1 + alpha + c**2/2
    foa2 = 4/a**2

    avec = [(m+n+1)/2, (m+n)/2+1, n+1, m+1]

    alpha_a = (alpha/a)**2
    One_a = (1/a)**2

    
    res =  mp.appellf4(avec[0], avec[1], avec[2], avec[3], alpha_a, One_a) * mp.binomial(m+n, n)*mp.power(2, -m-n - 1) / a *(alpha/a)**n * (1/a)**m 

    return res

def eval_lgf_exp(c, n, m, alpha, n_terms= 100):
    lbd = 2 + 2*alpha
    c_ratio = lbd/(lbd+c**2)
    a = 1 + alpha + c**2/2
    foa2 = 4/a**2
    res = 0

    for k in range(0, n_terms):
        diff = k - n - m
        if diff < 0:
            continue
        if diff % 2 != 0:
            continue
        for l in range(0, diff//2+1):
            #tmp = math.lgamma(k+1) - math.lgamma(l+1) - math.lgamma(n+l+1) - math.lgamma((diff - 2*l - n + m)//2+1) - math.lgamma((diff - 2*l - n - m)//2+1) + k*np.log(c_ratio/lbd) + (n+2*l)*np.log(alpha)
            tmp = mp.loggamma(k+1) - mp.loggamma(l+1) - mp.loggamma(n+l+1) - mp.loggamma((k - 2*l - n + m)//2+1) - mp.loggamma((k - 2*l - n - m)//2+1) + k*mp.log(c_ratio/lbd) + (n+2*l)*mp.log(alpha)
            #res += mp.binomial(k, n+2*l)*mp.binomial(n+2*l, l)*mp.binomial(diff - 2*l - n, (diff - 2*l - n + m)/2)*mp.power(c_ratio/lbd, k)*mp.power(alpha, n+2*l)
            res += mp.exp(tmp)
        #res += mp.binomial(k, )*mp.binomial(n, k)*mp.power(c_ratio, k)*mp.power(1-c_ratio, n-k)*mp.power(foa2, k)
    return res/(lbd + c**2)

def integrand(lbd, c, n,m,alpha,theta):
    #theta = np.linspace(-np.pi, np.pi, n_pts + 1, endpoint=True)
    a = alpha*np.cos(theta)*(0-2) + lbd + c**2
    K = (a + np.sqrt(np.square(a) - 4))/2
    res = ( np.cos(theta*n) * (1/K)**m) / (K - 1/K)
    return res

def eval_lgf_GK_finite(c, n,m, alpha, eps = 1e-10):
    #Evaluating the LGF using Gauss-Kronrod quadrature but with the 1D finite integral
    lbd = 2 + 2*alpha
    res = scipy.integrate.quad(lambda x: integrand(lbd, c, n,m,alpha,x), -np.pi, np.pi, epsrel=-1, epsabs=eps, limit = 100000)
    return res[0]/2/np.pi

def Bessel_representation(c, n, m, alpha, x_up = 10):
    lbd = 2 + 2*alpha
    coeffs = (lbd + c**2)/2
    res = scipy.integrate.quad(lambda x: (1j)**(n+m + 1)*np.exp(-coeffs*x*1j)*scipy.special.jv(n,alpha*x)*scipy.special.jv(m,x), 0, x_up, epsrel=1e-30, epsabs=1e-30, limit = 100000)
    print(coeffs)
    print(res)
    return res[0]/2

def Bessel_representation2(c, n, m, alpha, x_up = 10):
    lbd = 2 + 2*alpha
    coeffs = (lbd + c**2)/2
    res = scipy.integrate.quad(lambda x: (1j)**(n+m + 1)*np.sqrt(x)*scipy.special.jv(n,alpha*x)*scipy.special.jv(m,x) * (scipy.special.jv(-0.5,coeffs*x) - 1j*scipy.special.jv(0.5,coeffs*x)), 0, x_up, epsrel=1e-30, epsabs=1e-30, limit = 100000)
    print(coeffs)
    print(res)
    return res[0]/4*np.sqrt(np.pi*lbd)

def Bessel_representation3(c, n, m, alpha, x_up = 10):
    #From Duffin and Shelly 1958 (doesn't really work either)
    lbd = 2 + 2*alpha
    coeffs = (lbd + c**2)/2
    res = scipy.integrate.quad(lambda x: (1j)**(n+m)*np.exp(lbd*x + c**2*x)*scipy.special.iv(n,2*1j*alpha*x)*scipy.special.iv(m,2*1j*x), 0, x_up, epsrel=1e-30, epsabs=1e-30, limit = 100000)
    print(coeffs)
    print(res)
    return res[0]

def Bessel_representation4(c, n, m, alpha, x_up = 10, eps = 1e-12, limits = 50):
    #Maradudin
    res = scipy.integrate.quad(lambda x: np.exp(-c**2*x)*scipy.special.ive(n,2*alpha*x)*scipy.special.ive(m,2*x), 0, x_up, epsrel=-1, epsabs=eps, limit = limits)
    return res[0]

def Bessel_representation_pts(c, n, m, alpha, x_up = 10, eps = 1e-12):
    #Maradudin
    lbd = 2 + 2*alpha
    coeffs = (lbd + c**2)/2
    res = scipy.integrate.quad(lambda x: np.exp(-c**2*x)*scipy.special.ive(n,2*alpha*x)*scipy.special.ive(m,2*x), 0, x_up, epsrel=-1, epsabs=eps, points = [0.0000000001], limit = 1000)
    return res[0]

def eval_lgf_rfft(c, alpha, n, n_pts = 1e10):
    theta = np.linspace(0, np.pi, n_pts, endpoint=False, dtype=np.longdouble)

    
    #theta = np.linspace(0, np.pi, n_pts, endpoint=False)
    lbd = 2 + 2*alpha
    a = alpha*np.cos(theta)*(0-2) + lbd + c**2
    K = (a + np.sqrt(np.square(a) - 4))/2
    I = (1/K)**n / (K - 1/K)

    res = np.fft.irfft(I, (2*n_pts))
    #res = np.fft.fft(I)*2*np.pi * (n_pts * 2 - 1) / (n_pts * 2)


    m_vec = np.arange(0, len(res), 1, dtype=np.longdouble)

    a_pi = alpha*np.cos(np.pi)*(0-2) + lbd + c**2
    K_pi = (a_pi + np.sqrt(np.square(a_pi) - 4))/2
    I_pi = (1/K_pi)**n / (K_pi - 1/K_pi)
    
    res += I_pi / (n_pts * 2) * np.power(-1, m_vec)
    return res
