# 证明 $\frac{d}{dx}e^x=e^x$


```math

{\Huge

\begin{aligned}

\frac{d}{dx}e^x
&=\lim_{h \to 0}\frac{e^{x+h} - e^x}{h} \\
&=\lim_{h \to 0}\frac{e^{x} \cdot e^{h} - e^x}{h} \\
&=\lim_{h \to 0}e^x\frac{e^{h} - 1}{h} \\
&=e^x\lim_{h \to 0}\frac{e^{h} - 1}{h} \quad (e^x \text{ 为常数})\\
&=e^x\lim_{h \to 0}\frac{h}{h} \quad (\text{Taylor 一阶展开 } e^h \approx 1+h)\\
&=e^x

\end{aligned}

}

```
