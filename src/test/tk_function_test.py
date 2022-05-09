import tkinter as tk
from tkinter import ttk

window = tk.Tk()
window.title('GridSmart Detection')
window.geometry('1000x800')

# Style
style = ttk.Style()
style.configure("BW.TLabel", foreground = "black", background = 'white', font = ('Arial', 15))

# label
l_1 = tk.Label(window, text='test', font = ('Arial', 15), width = 30, height = 5)
l_1.pack()

ll = ttk.Label(text = 'ttk lable', style = "BW.TLabel")
ll.pack()

# text label
var_1 = tk.StringVar()
l_2 = tk.Label(window, textvariable=var_1, bg='green', fg='white', font=('Arial', 12), width=30, height=2)
l_2.pack()
# button
on_hit = False
def button_1_fun():
    global on_hit
    if on_hit == False:
        on_hit = True
        var_1.set('hit')
    else:
        on_hit = False
        var_1.set('')
button_1 = tk.Button(window, text = 'Button 1', font=('Arial', 12), width=10, height=1, command = button_1_fun)
button_1.pack()

# Entry
entry_1 = tk.Entry(window, show=None, font=('Arial', 14))
entry_1.pack()
# button
def fun_1():
    text = entry_1.get()
    text_1.insert('insert', text)

button_2 = tk.Button(window, text = 'Insert', width=10, height=2, command=fun_1)
button_2.pack()
# text
text_1 = tk.Text(window, height = 3)
text_1.pack()

# List
var_2 = tk.StringVar()
l_3 = tk.Label(window, bg='green', fg='yellow',font=('Arial', 12), width=10, textvariable=var_2)
l_3.pack()

def print_selection():
    value = lb.get(lb.curselection())
    var_2.set(value)

button_3 = tk.Button(window, text='print selection', width=15, height=2, command=print_selection)
button_3.pack()

var_3 = tk.StringVar()
var_3.set((1, 2, 3, 4))

lb = tk.Listbox(window, listvariable=var_3)
list_items = [11, 22, 33, 44]
for item in list_items:
    lb.insert('end', item)
lb.insert(1, 'first')
lb.pack()


#
window.mainloop()