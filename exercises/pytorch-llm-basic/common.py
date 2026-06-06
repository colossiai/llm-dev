"""Shared matplotlib configuration: enable Chinese fonts and fix minus sign."""

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Heiti TC",
    "PingFang HK",
    "Arial Unicode MS",
    "Songti SC",
    "STHeiti",
]
plt.rcParams["axes.unicode_minus"] = False
