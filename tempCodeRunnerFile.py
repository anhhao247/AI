x = np.linspace(-5, 5, 50)
y = exp(x)

plt.plot(x, y)
plt.title("Đồ thị hàm số y = exp(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()