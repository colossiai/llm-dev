# 介绍 ln(x)


`ln(x) 表示“要把 e 提升到多少次方，才能得到 x` : 

如果 $\ln(x) = y$

那么 $e^y$ = x, 即 $e^{ln(x)} = x$


## 证明 ln(ab) = ln(a) + ln(b)




```math

{\Huge

\begin{aligned}

e^{\ln(ab)}
&=ab\\
&=e^{ln(a)} \cdot e^{ln(b)}\\
&=e^{ln(a) + ln(b)}\\

\end{aligned}

}

```

所以

```math
{\Huge

\begin{aligned}

\ln(ab) = ln(a) + ln(b)

\end{aligned}

}

```

同理

```math
{\Huge

\begin{aligned}

\ln(a/b) = ln(a) - ln(b)

\end{aligned}

}

```


## 求 $\frac{d}{dx}a^x$


```math
{\Huge

\begin{aligned}

\frac{d}{dx}a^x
&=\frac{d}{dx}e^{\ln(a^x)} \quad (\text{因为} u = e^{\ln(u)})\\
&=\frac{d}{dx}e^{x\ln(a)}  \quad (\ln(a^x) = x\ln(a))\\
\\
\text{Let } y = x\ln(a) \\
\\
\frac{d}{dx}e^{x\ln(a)} 
&=\frac{d}{dx}e^y \\
&=\frac{d(e^y)}{d(y)} \cdot \frac{d(y)}{d(x)} \quad (\text{Chain Rule})\\
&=e^y \cdot \ln(a)  \quad (e^y\text{导数为自己, d(y)/d(x)导数为常数ln(a)})\\
&=e^{x\ln(a)} \cdot \ln(a) \\
&=e^{\ln(a^x)} \cdot \ln(a) \\
&=a^x \cdot \ln(a) \\


\end{aligned}

}

```




## 求 $\frac{d}{dx}\ln(x)$


${\Huge \text{Let } y=ln(x), x=e^y}$

${\Huge \frac{d}{dx}(x)=\frac{d}{dx}(e^y) }$

${\Huge \text{So: } 1=\frac{d}{dx}(e^y)  }$


```math
{\Huge

\begin{aligned}

1
&=\frac{d}{dx}(e^y) \\
&=\frac{d}{dy}(e^y) \cdot \frac{d}{dx}(y) \quad  (\text{Chain Rule}) \\
&=e^y \cdot \frac{d}{dx}(y) \\
&=x \cdot \frac{d}{dx}(y) \quad ( e^y = x )\\


\end{aligned}
}

```

${\Huge \text{So: } \frac{d}{dx}(y) = \frac{d}{dx}\ln(x) = \frac{1}{x} }$


## 求 $\frac{d}{dx}\log_a x$


${\Huge \text{Let } log_a(x) = y, a^y = x}$

${\Huge \frac{d}{dx}(x)=\frac{d}{dx}(a^y) }$

${\Huge \text{So: } 1=\frac{d}{dx}(a^y)  }$





```math
{\Huge

\begin{aligned}

1
&=\frac{d}{dx}(a^y) \\
&=\frac{d}{dy}(a^y) \cdot \frac{d}{dx}(y) \quad  (\text{Chain Rule}) \\
&=a^y\ln(a) \cdot \frac{d}{dx}(y) \\
&=x\ln(a) \cdot \frac{d}{dx}(y) \quad ( a^y = x )\\


\end{aligned}
}

```

${\Huge \text{So: } \frac{d}{dx}(y) = \frac{d}{dx}\log_a(x) = \frac{1}{x\ln(a)} }$