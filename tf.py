import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sympy import symbols
import math as m

st.markdown("<h1 style='text-align: center; color: white;font-size: 60px'>Triangular Function</h1>",
            unsafe_allow_html=True)

st.title("Input 1 (Dirt)")

input1 = st.slider('Select the value for Input 1 (Dirt)', 0, 100, 50)

x1 = np.array([50, 0])
y1 = np.array([0, 1])
x2 = np.array([0, 100, 50, 0])
y2 = np.array([0, 0, 1, 0])
x3 = np.array([50, 100])
y3 = np.array([0, 1])

fig1 = plt.figure(1, figsize=(12, 7))
ax1 = fig1.add_subplot(111)
# plt.subplots_adjust(right=1)

plt.plot(x1, y1)
plt.text(5, 0.9, 'Little')
plt.plot(x2, y2)
plt.text(47, 0.9, 'Medium')
plt.plot(x3, y3)
plt.text(91, 0.9, 'High')

if input1 < 50:
    plt.fill("j", "k", 'm', data={"j": [50, 0, 0], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [0, 50, 50], "k": [0, 1, 0]})
elif input1 == 50:
    plt.fill("j", "k", 'm', data={"j": [50, 0, 0], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [0, 50, 50], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [50, 50, 100], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [50, 100, 100], "k": [0, 1, 0]})
elif input1 > 50 and input1 <= 100:
    plt.fill("j", "k", 'm', data={"j": [50, 50, 100], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [50, 100, 100], "k": [0, 1, 0]})


plt.xlim(0, 100)
plt.ylim(0, 1)

ax1.set_xlabel('Dirt')
ax1.set_ylabel('Membership Function')

plt.grid(axis='both', which='major', color=[
         166/255, 166/255, 166/255], linestyle='-', linewidth=2)
plt.minorticks_on()
plt.grid(axis='both', which='minor', color=[
         166/255, 166/255, 166/255], linestyle=':', linewidth=2)

st.pyplot(fig1)

st.title("Input 2 (Grease)")

input2 = st.slider('Select the value for Input 2 (Grease)', 0, 100, 50)

x1 = np.array([50, 0])
y1 = np.array([0, 1])
x2 = np.array([0, 100, 50, 0])
y2 = np.array([0, 0, 1, 0])
x3 = np.array([50, 100])
y3 = np.array([0, 1])

fig2 = plt.figure(2, figsize=(12, 7))
ax1 = fig2.add_subplot(111)
# plt.subplots_adjust(right=1)

plt.plot(x1, y1)
plt.text(5, 0.9, 'Little')
plt.plot(x2, y2)
plt.text(47, 0.9, 'Medium')
plt.plot(x3, y3)
plt.text(91, 0.9, 'High')

if input2 < 50:
    plt.fill("j", "k", 'm', data={"j": [50, 0, 0], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [0, 50, 50], "k": [0, 1, 0]})
elif input2 == 50:
    plt.fill("j", "k", 'm', data={"j": [50, 0, 0], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [0, 50, 50], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [50, 50, 100], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [50, 100, 100], "k": [0, 1, 0]})
elif input2 > 50 and input2 <= 100:
    plt.fill("j", "k", 'm', data={"j": [50, 50, 100], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [50, 100, 100], "k": [0, 1, 0]})

plt.xlim(0, 100)
plt.ylim(0, 1)

ax1.set_xlabel('Grease')
ax1.set_ylabel('Membership Function')

plt.grid(axis='both', which='major', color=[
         166/255, 166/255, 166/255], linestyle='-', linewidth=2)
plt.minorticks_on()
plt.grid(axis='both', which='minor', color=[
         166/255, 166/255, 166/255], linestyle=':', linewidth=2)

st.pyplot(fig2)

st.title("Rule Base")

data = [{'Little Grease': 'Very Less (Time)', 'Medium Grease': 'Medium (Time)', 'High Grease': 'High (Time)'}, {'Little Grease': 'Less (Time)', 'Medium Grease': 'High (Time)',
                                                                                                                'High Grease': 'Very High (Time)'}, {'Little Grease': 'Medium (Time)', 'Medium Grease': 'High (Time)', 'High Grease': 'Very High (Time)'}]


df = pd.DataFrame(data, index=['Little Dirt',
                  'Medium Dirt', 'High Dirt'])

if input1 < 50:
    if input2 < 50:
        st.dataframe(df.style.apply(lambda x: ['background: magenta' if (i == 'Little Dirt' and x.name == 'Little Grease') or (i == 'Little Dirt' and x.name == 'Medium Grease') or (i == 'Medium Dirt' and x.name == 'Little Grease') or (i == 'Medium Dirt' and x.name == 'Medium Grease') else ''
                                               for i, _ in x.iteritems()]))
    if input2 > 50:
        st.dataframe(df.style.apply(lambda x: ['background: magenta' if (i == 'Little Dirt' and x.name == 'Medium Grease') or (i == 'Little Dirt' and x.name == 'High Grease') or (i == 'Medium Dirt' and x.name == 'Medium Grease') or (i == 'Medium Dirt' and x.name == 'High Grease') else ''
                                               for i, _ in x.iteritems()]))

if input1 > 50:
    if input2 < 50:
        st.dataframe(df.style.apply(lambda x: ['background: magenta' if (i == 'Medium Dirt' and x.name == 'Little Grease') or (i == 'Medium Dirt' and x.name == 'Medium Grease') or (i == 'High Dirt' and x.name == 'Little Grease') or (i == 'High Dirt' and x.name == 'Medium Grease') else ''
                                               for i, _ in x.iteritems()]))
    elif input2 > 50:
        st.dataframe(df.style.apply(lambda x: ['background: magenta' if (i == 'Medium Dirt' and x.name == 'Medium Grease') or (i == 'Medium Dirt' and x.name == 'High Grease') or (i == 'High Dirt' and x.name == 'Medium Grease') or (i == 'High Dirt' and x.name == 'High Grease') else ''
                                               for i, _ in x.iteritems()]))

if input1 == 50 and input2 == 50:
    st.dataframe(df.style.apply(lambda x: ['background: magenta' if x.name == 'Little Grease' or x.name ==
                                           'Medium Grease' or x.name == 'High Grease' else '' for i in x]))

if input1 == 50:
    if input2 < 50:
        st.dataframe(df.style.apply(lambda x: ['background: magenta' if (i == 'Little Dirt' and x.name == 'Little Grease') or (i == 'Little Dirt' and x.name == 'Medium Grease') or (i == 'Medium Dirt' and x.name == 'Little Grease') or (i == 'Medium Dirt' and x.name == 'Medium Grease') or (i == 'High Dirt' and x.name == 'Little Grease') or (i == 'High Dirt' and x.name == 'Medium Grease') else ''
                                               for i, _ in x.iteritems()]))
    elif input2 > 50:
        st.dataframe(df.style.apply(lambda x: ['background: magenta' if (i == 'Little Dirt' and x.name == 'Medium Grease') or (i == 'Little Dirt' and x.name == 'High Grease') or (i == 'Medium Dirt' and x.name == 'Medium Grease') or (i == 'Medium Dirt' and x.name == 'High Grease') or (i == 'High Dirt' and x.name == 'Medium Grease') or (i == 'High Dirt' and x.name == 'High Grease') else ''
                                               for i, _ in x.iteritems()]))

if input2 == 50:
    if input1 < 50:
        st.dataframe(df.style.apply(lambda x: ['background: magenta' if (i == 'Little Dirt' and x.name == 'Little Grease') or (i == 'Medium Dirt' and x.name == 'Little Grease') or (i == 'Little Dirt' and x.name == 'Medium Grease') or (i == 'Medium Dirt' and x.name == 'Medium Grease') or (i == 'Little Dirt' and x.name == 'High Grease') or (i == 'Medium Dirt' and x.name == 'High Grease') else ''
                                               for i, _ in x.iteritems()]))
    elif input1 > 50:
        st.dataframe(df.style.apply(lambda x: ['background: magenta' if (i == 'Medium Dirt' and x.name == 'Little Grease') or (i == 'High Dirt' and x.name == 'Little Grease') or (i == 'Medium Dirt' and x.name == 'Medium Grease') or (i == 'High Dirt' and x.name == 'Medium Grease') or (i == 'Medium Dirt' and x.name == 'High Grease') or (i == 'High Dirt' and x.name == 'High Grease') else ''
                                               for i, _ in x.iteritems()]))


st.title("Output (Rule Base)")

x1 = np.array([10, 0])
y1 = np.array([0, 1])
x2 = np.array([0, 12.5, 25])
y2 = np.array([0, 1, 0])
x3 = np.array([10, 25, 40])
y3 = np.array([0, 1, 0])
x4 = np.array([25, 42.5, 60])
y4 = np.array([0, 1, 0])
x5 = np.array([40, 60])
y5 = np.array([0, 1])

fig3 = plt.figure(3, figsize=(12, 7))
ax1 = fig3.add_subplot(111)
# plt.subplots_adjust(right=1)

plt.plot(x1, y1)
plt.text(1, 0.9, 'Very Less')
plt.plot(x2, y2)
plt.text(11.5, 0.9, 'Less')
plt.plot(x3, y3)
plt.text(23.2, 0.9, 'Medium')
plt.plot(x4, y4)
plt.text(41.5, 0.9, 'High')
plt.plot(x5, y5)
plt.text(53.3, 0.9, 'Very High')

if input1 < 50:
    if input2 < 50:
        plt.fill("j", "k", 'm', data={"j": [10, 0, 0], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [10, 25, 40], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [0, 12.5, 25], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
    if input2 > 50:
        plt.fill("j", "k", 'm', data={"j": [10, 25, 40], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [40, 60, 60], "k": [0, 1, 0]})

if input1 > 50:
    if input2 < 50:
        plt.fill("j", "k", 'm', data={"j": [0, 12.5, 25], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [10, 25, 40], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
    elif input2 > 50:
        plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [40, 60, 60], "k": [0, 1, 0]})

if input1 == 50 and input2 == 50:
    plt.fill("j", "k", 'm', data={"j": [10, 0, 0], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [10, 25, 40], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [0, 12.5, 25], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
    plt.fill("j", "k", 'm', data={"j": [40, 60, 60], "k": [0, 1, 0]})

if input1 == 50:
    if input2 < 50:
        plt.fill("j", "k", 'm', data={"j": [10, 0, 0], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [10, 25, 40], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [0, 12.5, 25], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
    elif input2 > 50:
        plt.fill("j", "k", 'm', data={"j": [10, 25, 40], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [40, 60, 60], "k": [0, 1, 0]})

if input2 == 50:
    if input1 < 50:
        plt.fill("j", "k", 'm', data={"j": [10, 0, 0], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [10, 25, 40], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [0, 12.5, 25], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [40, 60, 60], "k": [0, 1, 0]})
    elif input1 > 50:
        plt.fill("j", "k", 'm', data={"j": [10, 25, 40], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [0, 12.5, 25], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
        plt.fill("j", "k", 'm', data={"j": [40, 60, 60], "k": [0, 1, 0]})


plt.xlim(0, 60)
plt.ylim(0, 1)

ax1.set_xlabel('Time')
ax1.set_ylabel('Membership Function')

plt.grid(axis='both', which='major', color=[
         166/255, 166/255, 166/255], linestyle='-', linewidth=2)
plt.minorticks_on()
plt.grid(axis='both', which='minor', color=[
         166/255, 166/255, 166/255], linestyle=':', linewidth=2)

st.pyplot(fig3)

sd = (50-input1)/50
m1 = input1/50
m2 = (100-input1)/50
md = min(m1, m2)
hd = (input1-50)/50

sg = (50-input2)/50
m3 = input2/50
m4 = (100-input2)/50
mg = min(m3, m4)
hg = (input2-50)/50

if input1 < 50:
    if input2 < 50:
        op = max(min(sd, sg), min(sd, mg), min(md, sg), min(md, mg))
    if input2 > 50:
        op = max(min(sd, mg), min(sd, hg), min(md, mg), min(md, hg))

if input1 > 50:
    if input2 < 50:
        op = max(min(md, sg), min(md, mg), min(hd, sg), min(hd, mg))
    elif input2 > 50:
        op = max(min(md, mg), min(md, hg), min(hd, mg), min(hd, hg))

if input1 == 50 and input2 == 50:
    op = max(min(sd, sg), min(sd, mg), min(sd, hg), min(md, sg), min(
        md, mg), min(md, hg), min(hd, sg), min(hd, mg), min(hd, hg))

if input1 == 50:
    if input2 < 50:
        op = max(min(sd, sg), min(sd, mg), min(md, sg),
                 min(md, mg), min(hd, sg), min(hd, mg))
    elif input2 > 50:
        op = max(min(sd, mg), min(sd, hg), min(md, mg),
                 min(md, hg), min(hd, mg), min(hd, hg))

if input2 == 50:
    if input1 < 50:
        op = max(min(sd, sg), min(sd, mg), min(sd, hg),
                 min(md, sg), min(md, mg), min(md, hg))
    elif input1 > 50:
        op = max(min(md, sg), min(md, mg), min(md, hg),
                 min(hd, sg), min(hd, mg), min(hd, hg))

st.title("Output (Rule Strength)")

if op == min(sd, sg):
    st.write("The output is in (Little Dirt, Little Grease) i.e. Very Less Time")
if op == min(sd, mg):
    st.write("The output is in (Little Dirt, Medium Grease) i.e. Medium Time")
if op == min(sd, hg):
    st.write("The output is in (Little Dirt, High Grease) i.e. High Time")
if op == min(md, sg):
    st.write("The output is in (Medium Dirt, Little Grease) i.e. Less Time")
if op == min(md, mg):
    st.write("The output is in (Medium Dirt, Medium Grease) i.e. High Time")
if op == min(md, hg):
    st.write("The output is in (Medium Dirt, High Grease) i.e. Very High Time")
if op == min(hd, sg):
    st.write("The output is in (High Dirt, Little Grease) i.e. Medium Time")
if op == min(hd, mg):
    st.write("The output is in (High Dirt, Medium Grease) i.e. High Time")
if op == min(hd, hg):
    st.write("The output is in (High Dirt, High Grease) i.e. Very High Time")

x1 = np.array([10, 0])
y1 = np.array([0, 1])
x2 = np.array([0, 12.5, 25])
y2 = np.array([0, 1, 0])
x3 = np.array([10, 25, 40])
y3 = np.array([0, 1, 0])
x4 = np.array([25, 42.5, 60])
y4 = np.array([0, 1, 0])
x5 = np.array([40, 60])
y5 = np.array([0, 1])

fig4 = plt.figure(4, figsize=(12, 7))
ax1 = fig4.add_subplot(111)
# plt.subplots_adjust(right=1)

plt.plot(x1, y1)
plt.text(1, 0.9, 'Very Less')
plt.plot(x2, y2)
plt.text(11.5, 0.9, 'Less')
plt.plot(x3, y3)
plt.text(23.2, 0.9, 'Medium')
plt.plot(x4, y4)
plt.text(41.5, 0.9, 'High')
plt.plot(x5, y5)
plt.text(53.3, 0.9, 'Very High')

if op == min(sd, sg):
    plt.fill("j", "k", 'm', data={"j": [10, 0, 0], "k": [0, 1, 0]})
if op == min(sd, mg):
    plt.fill("j", "k", 'm', data={"j": [10, 25, 40], "k": [0, 1, 0]})
if op == min(sd, hg):
    plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
if op == min(md, sg):
    plt.fill("j", "k", 'm', data={"j": [0, 12.5, 25], "k": [0, 1, 0]})
if op == min(md, mg):
    plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
if op == min(md, hg):
    plt.fill("j", "k", 'm', data={"j": [40, 60, 60], "k": [0, 1, 0]})
if op == min(hd, sg):
    plt.fill("j", "k", 'm', data={"j": [10, 25, 40], "k": [0, 1, 0]})
if op == min(hd, mg):
    plt.fill("j", "k", 'm', data={"j": [25, 42.5, 60], "k": [0, 1, 0]})
if op == min(hd, hg):
    plt.fill("j", "k", 'm', data={"j": [40, 60, 60], "k": [0, 1, 0]})

plt.xlim(0, 60)
plt.ylim(0, 1)

ax1.set_xlabel('Time')
ax1.set_ylabel('Membership Function')

plt.grid(axis='both', which='major', color=[
         166/255, 166/255, 166/255], linestyle='-', linewidth=2)
plt.minorticks_on()
plt.grid(axis='both', which='minor', color=[
         166/255, 166/255, 166/255], linestyle=':', linewidth=2)

st.pyplot(fig4)

# st.title("Defuzzified Output")

# if op == min(sd, sg):
#     pass
# if op == min(sd, mg):
#     x, y = symbols('x y')
#     l1 = (15*x)+10
#     l2 = 40-(15*y)
#     a = l1.subs(x, op)
#     b = l2.subs(y, op)
#     fop = (a+b)/2
#     if fop-int(fop) < 0.5:
#         fop = m.floor(fop)
#     else:
#         fop = m.ceil(fop)
#     st.write(f'The time required for cleaning is {fop} minutes')
# if op == min(sd, hg):
#     x, y = symbols('x y')
#     l1 = (17.5*x)+25
#     l2 = 60-(17.5*y)
#     a = l1.subs(x, op)
#     b = l2.subs(y, op)
#     fop = (a+b)/2
#     if fop-int(fop) < 0.5:
#         fop = m.floor(fop)
#     else:
#         fop = m.ceil(fop)
#     st.write(f'The time required for cleaning is {fop} minutes')
# if op == min(md, sg):
#     x, y = symbols('x y')
#     l1 = 12.5*x
#     l2 = 25-(12.5*y)
#     a = l1.subs(x, op)
#     b = l2.subs(y, op)
#     fop = (a+b)/2
#     if fop-int(fop) < 0.5:
#         fop = m.floor(fop)
#     else:
#         fop = m.ceil(fop)
#     st.write(f'The time required for cleaning is {fop} minutes')
# if op == min(md, mg):
#     x, y = symbols('x y')
#     l1 = (17.5*x)+25
#     l2 = 60-(17.5*y)
#     a = l1.subs(x, op)
#     b = l2.subs(y, op)
#     fop = (a+b)/2
#     if fop-int(fop) < 0.5:
#         fop = m.floor(fop)
#     else:
#         fop = m.ceil(fop)
#     st.write(f'The time required for cleaning is {fop} minutes')
# if op == min(md, hg):
#     pass
# if op == min(hd, sg):
#     x, y = symbols('x y')
#     l1 = (15*x)+10
#     l2 = 40-(15*y)
#     a = l1.subs(x, op)
#     b = l2.subs(y, op)
#     fop = (a+b)/2
#     if fop-int(fop) < 0.5:
#         fop = m.floor(fop)
#     else:
#         fop = m.ceil(fop)
#     st.write(f'The time required for cleaning is {fop} minutes')
# if op == min(hd, mg):
#     x, y = symbols('x y')
#     l1 = (17.5*x)+25
#     l2 = 60-(17.5*y)
#     a = l1.subs(x, op)
#     b = l2.subs(y, op)
#     fop = (a+b)/2
#     if fop-int(fop) < 0.5:
#         fop = m.floor(fop)
#     else:
#         fop = m.ceil(fop)
#     st.write(f'The time required for cleaning is {fop} minutes')
# if op == min(hd, hg):
#     pass
