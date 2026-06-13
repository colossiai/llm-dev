# sigmoid求导


${\Huge S(x)=\frac{1}{1+e^{-x}} }$

${\Huge \text{Let } y=1+e^{-x} \text{, so } S(x) = y^{-1}}$

```math

{\Huge

\begin{aligned}

\frac{d}{dx}S(x)
&=\frac{d}{dy}S(x) \cdot \frac{dy}{dx} \quad (\text {Chain Rule})\\
&=-1(y)^{-2} \cdot (0 + \frac{d}{dx}e^{-x})\\
&=-1(y)^{-2} \cdot \frac{d}{dx}e^{-x}\\
&=-1(y)^{-2} \cdot (\frac{d}{du}e^u \cdot \frac{d}{dx}u) \quad (\text {Let } u = -x) \\
&=-1(y)^{-2} \cdot (e^u \cdot -1)\\
&=-1(y)^{-2} \cdot (e^{-x} \cdot -1)\\
&=\frac{e^{-x}}{y^2}\\
&=\frac{y-1}{y^2}\\
&=\frac{1}{y} \cdot (1 - \frac{1}{y}) \\
&=S(x) \cdot (1 - S(x))\\

\end{aligned}

}

```
