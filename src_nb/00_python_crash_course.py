# %% [markdown]
"""
# 00 · Python crash course for chemists

**Chemoinformatics practicals — Session 0 of 9**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

> **How to use this notebook.** Click the *Open in Colab* badge above, then run the cells one by one
> with `Shift + Enter`. Nothing is installed on your computer: everything runs in the cloud.
> To keep your work, choose *File → Save a copy in Drive*.

**Learning goals.** After this session you will be able to
- run Python code in a Jupyter / Google Colab notebook and read an error message;
- use variables, numbers, strings, lists, dictionaries, loops, conditions and functions;
- use `numpy` for arrays, `matplotlib` for plots and `pandas` for tables;
- load a real chemical dataset (aqueous solubility of ~1100 compounds) and explore it.

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - *AI for Chemistry* (EPFL CH-457) by the **Schwaller group** — `01a_python_crash_course`, `01b_python_essentials_pandas` ([GitHub](https://github.com/schwallergroup/ai4chem_course), MIT license);
> - *Practical Programming in Chemistry* (EPFL CH-200) by the **Schwaller group** ([GitHub](https://github.com/schwallergroup/practical-programming-in-chemistry-exercises), MIT license);
> - **MolSSI** cheminformatics workshop by Jessica A. Nash — `00_python_basics` ([GitHub](https://github.com/MolSSI-Education/molssi-cheminformatics), MIT license);
> - *Practical Cheminformatics Tutorials* by **Pat Walters** — `pandas_intro` ([GitHub](https://github.com/PatWalters/practical_cheminformatics_tutorials), MIT license).
>
> The ESOL solubility dataset is from Delaney, *J. Chem. Inf. Comput. Sci.* **2004**, 44, 1000 (via [DeepChem/MoleculeNet](https://github.com/deepchem/deepchem)).
"""

# %% [markdown]
"""
## 0. Jupyter notebooks and Google Colab

A notebook is a sequence of **cells**. There are two kinds:

- **Markdown cells** (like this one) contain formatted text, equations ($\Delta G = \Delta H - T\Delta S$) and images.
- **Code cells** contain Python code. When you run a code cell, its output appears right below it.

Useful shortcuts in Colab:

| action | shortcut |
|---|---|
| run the current cell and move to the next | `Shift + Enter` |
| run the current cell and stay | `Ctrl + Enter` |
| insert a code cell below | `Ctrl + M`, then `B` |
| turn a cell into Markdown | `Ctrl + M`, then `M` |
| show the documentation of a function | put the cursor inside the parentheses and press `Tab` (or `Shift + Tab` in Jupyter) |

Cells share one **kernel** (a running Python process). The order in which you *run* cells matters, not the
order in which they appear. If things get confusing, use *Runtime → Restart session* and run from the top.
"""

# %%
# This is a code cell. Lines that begin with '#' are comments and are ignored by Python.
# Run me with Shift + Enter.
print("Hello, chemoinformatics!")

# %% [markdown]
"""
## 1. Python as a calculator: numbers and variables

Let's compute the Gibbs free energy of a reaction, $\Delta G = \Delta H - T\,\Delta S$.
"""

# %%
delta_H = -541.5   # kJ/mol
delta_S = 10.4     # J/(mol K)   <- careful with units!
T = 298.15         # K

delta_G = delta_H - T * delta_S / 1000   # convert J -> kJ
print(delta_G)

# %% [markdown]
"""
A few things to notice:

- `delta_H = -541.5` **assigns** the value on the right to the *variable* on the left. The variable name is
  your choice (letters, digits, underscores; it cannot start with a digit; `deltaH` and `deltah` are different).
- `print(...)` is a **function**: a named piece of code that does something with the *arguments* in parentheses.
- Arithmetic works as you expect: `+ - * /`, `**` for powers, `//` for integer division and `%` for the remainder.

If the last line of a cell is an expression, the notebook displays its value even without `print`:
"""

# %%
delta_G

# %%
type(delta_G)        # every value has a type; here a floating-point number

# %%
n_atoms = 21         # an integer
name = "caffeine"    # a string (text) - single or double quotes both work
is_aromatic = True   # a boolean
print(type(n_atoms), type(name), type(is_aromatic))

# %% [markdown]
"""
### f-strings: putting values inside text

Formatted string literals (`f"..."`) are the easiest way to build readable output.
`{value:.2f}` means "show 2 decimals".
"""

# %%
print(f"ΔG at {T} K is {delta_G:.2f} kJ/mol")
print(f"{name} has {n_atoms} atoms; aromatic: {is_aromatic}")

# %% [markdown]
"""
### Exercise 1.1

The pH of a buffer is given by the Henderson–Hasselbalch equation $\mathrm{pH} = \mathrm{p}K_a + \log_{10}\frac{[A^-]}{[HA]}$.
Compute the pH of an acetate buffer (p$K_a$ = 4.76) with $[A^-]$ = 0.10 M and $[HA]$ = 0.05 M.
You need the `math` module: `import math` gives you `math.log10`.
"""

# %%
import math

# YOUR CODE HERE
pKa = 4.76
A_minus = 0.10
HA = 0.05
pH = ...  # replace the ... with the formula

# print(f"pH = {pH:.2f}")

# %% [markdown]
"""
<details><summary><b>Solution</b> (click to expand)</summary>

```python
pH = pKa + math.log10(A_minus / HA)
print(f"pH = {pH:.2f}")   # pH = 5.06
```
</details>
"""

# %% [markdown]
"""
## 2. Collections: lists, tuples, dictionaries, sets

Chemistry is full of *collections* of things: atoms in a molecule, compounds in a library, measurements in an assay.
"""

# %%
# A list is an ordered, changeable sequence. Indexing starts at 0!
energies_kcal = [-13.4, -2.7, 5.4, 42.1]
print(energies_kcal[0])       # first element
print(energies_kcal[-1])      # last element
print(energies_kcal[1:3])     # a "slice": elements 1 and 2 (the end index is excluded)
print(len(energies_kcal))     # number of elements

# %%
energies_kcal.append(-0.8)    # add an element at the end
energies_kcal[0] = -13.5      # modify an element
print(energies_kcal)
print(sorted(energies_kcal))  # a sorted copy
print(min(energies_kcal), max(energies_kcal), sum(energies_kcal))

# %%
# A tuple is like a list but cannot be modified. Often used for fixed records.
water = ("H2O", 18.015, 100.0)      # formula, molar mass, boiling point
formula, mass, bp = water           # "unpacking"
print(formula, mass, bp)

# %%
# A dictionary maps keys to values - perfect for lookup tables.
atomic_mass = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}
print(atomic_mass["C"])
atomic_mass["Cl"] = 35.45            # add a new entry
print("Cl" in atomic_mass, "Br" in atomic_mass)   # membership test
print(list(atomic_mass.keys()))

# %%
# A set is an unordered collection of unique elements - handy for deduplication.
elements_in_caffeine = {"C", "H", "N", "O", "C", "N"}
print(elements_in_caffeine)

# %% [markdown]
"""
### Strings are sequences too

Strings behave like lists of characters (indexing, slicing, `len`) and have many useful methods.
A SMILES string (we'll learn them next session) is just a string!
"""

# %%
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"   # caffeine
print(len(smiles), smiles[0], smiles[:5])
print(smiles.count("N"), "N atoms (roughly)")
print(smiles.lower())
print(smiles.replace("C", "c"))          # replace (returns a new string, original unchanged)
print("caffeine".upper(), "  spaces  ".strip(), "a,b,c".split(","))

# %% [markdown]
"""
## 3. Control flow: `if`, `for`, `while`

**Indentation matters in Python**: the block belonging to `if`/`for`/`def` is the set of lines indented by 4 spaces.
"""

# %%
logP = 3.2
if logP > 5:
    print("very lipophilic")
elif logP > 3:
    print("lipophilic")
else:
    print("hydrophilic")

# %%
# for loops iterate over any collection
for e in energies_kcal:
    e_kJ = e * 4.184
    print(f"{e:6.1f} kcal/mol = {e_kJ:7.1f} kJ/mol")

# %%
# The typical pattern: build a new list from an old one
energies_kJ = []
for e in energies_kcal:
    energies_kJ.append(e * 4.184)
print(energies_kJ)

# The same in one line: a "list comprehension" (very common in Python code you will read)
energies_kJ = [e * 4.184 for e in energies_kcal]
print(energies_kJ)

# Comprehensions can filter too
negative = [e for e in energies_kJ if e < 0]
print(negative)

# %%
# enumerate gives you the index and the value; zip walks through several lists together
formulas = ["H2O", "CO2", "NH3"]
masses = [18.015, 44.009, 17.031]
for i, (f, m) in enumerate(zip(formulas, masses)):
    print(i, f, m)

# %%
# Iterating over a dictionary
for symbol, m in atomic_mass.items():
    print(f"{symbol:>2s}: {m:6.3f}")

# %%
# while loops repeat until a condition becomes False - e.g. a simple titration model
volume_added = 0.0
pH = 3.0
while pH < 7.0:
    volume_added += 0.5
    pH += 0.4          # (a crude, linear toy model)
print(f"Neutral after adding {volume_added} mL")

# %% [markdown]
"""
### Exercise 3.1 — molecular weight from a formula dictionary

Caffeine is C$_8$H$_{10}$N$_4$O$_2$. Using the dictionary `atomic_mass` defined above and the dictionary
`caffeine = {"C": 8, "H": 10, "N": 4, "O": 2}`, compute its molecular weight with a `for` loop.
(Expected answer: ≈ 194.19 g/mol.)
"""

# %%
caffeine = {"C": 8, "H": 10, "N": 4, "O": 2}

# YOUR CODE HERE
mw = 0.0


# print(f"MW(caffeine) = {mw:.2f} g/mol")

# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
mw = 0.0
for element, count in caffeine.items():
    mw += count * atomic_mass[element]
print(f"MW(caffeine) = {mw:.2f} g/mol")   # 194.19
```
</details>
"""

# %% [markdown]
"""
## 4. Functions

When you do the same thing twice, write a **function**. A function has a name, *parameters*, a *docstring*
(documentation) and a `return` value.
"""

# %%
def molecular_weight(formula, masses=atomic_mass):
    """Return the molecular weight (g/mol) of a molecule given as {element: count}."""
    mw = 0.0
    for element, count in formula.items():
        mw += count * masses[element]
    return mw

print(molecular_weight(caffeine))
print(molecular_weight({"H": 2, "O": 1}))
aspirin = {"C": 9, "H": 8, "O": 4}
print(f"aspirin: {molecular_weight(aspirin):.2f}")

# %%
def gibbs(delta_H, delta_S, T=298.15):
    """Gibbs free energy in kJ/mol from ΔH (kJ/mol), ΔS (J/mol/K) and T (K, default 298.15)."""
    return delta_H - T * delta_S / 1000

print(gibbs(-541.5, 10.4))          # uses the default temperature
print(gibbs(-541.5, 10.4, T=350))   # keyword argument

# %%
help(gibbs)   # the docstring is what help() shows - write them!

# %% [markdown]
"""
### Exercise 4.1 — a classification function

Write a function `classify_logP(logP)` that returns the string `"hydrophilic"`, `"lipophilic"` or `"very lipophilic"`
(same thresholds as in section 3), then apply it to every value in `[−1.2, 0.5, 3.4, 6.1]` with a list comprehension.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
def classify_logP(logP):
    if logP > 5:
        return "very lipophilic"
    elif logP > 3:
        return "lipophilic"
    return "hydrophilic"

print([classify_logP(x) for x in [-1.2, 0.5, 3.4, 6.1]])
```
</details>
"""

# %% [markdown]
"""
## 5. Reading error messages

Errors are normal. Read the **last line** of the message first: it tells you the type of error and usually what went wrong.
Then look at the arrow (`---->`) showing which line caused it.
"""

# %%
# Remove the '#' in front of the next line, run the cell and read the message. Then fix the typo and run again.
# print(molecular_weight(caffein))

# %%
# Some frequent errors - uncomment one line at a time to see the message
# print(atomic_mass["Xx"])          # KeyError: the key does not exist
# print(energies_kcal[10])          # IndexError: list index out of range
# print("MW = " + 194.19)           # TypeError: cannot add str and float -> use f-strings
# print(1 / 0)                      # ZeroDivisionError
# import rdkitt                     # ModuleNotFoundError: the package is not installed (or misspelled)

# %%
# You can catch errors with try/except when you expect them (e.g. invalid input in a large dataset)
def safe_mass(symbol):
    try:
        return atomic_mass[symbol]
    except KeyError:
        print(f"Warning: unknown element {symbol!r}")
        return None

print(safe_mass("C"), safe_mass("Xx"))

# %% [markdown]
"""
## 6. Modules and packages: `numpy` and `matplotlib`

Python's power comes from its **packages**. You `import` them (optionally with a short alias) and then use their functions.

- `math` — basic maths (`sqrt`, `log10`, `exp`, `pi`)
- `numpy` (`np`) — fast numerical arrays
- `matplotlib.pyplot` (`plt`) — plots
- `pandas` (`pd`) — tables (next section)
- `rdkit` — chemistry (next session!)

On Colab, `numpy`, `pandas` and `matplotlib` are preinstalled. Other packages are installed with `!pip install package`.
"""

# %%
import numpy as np

energies = np.array(energies_kcal)          # a numpy array
print(energies * 4.184)                      # vectorised operation: no loop needed!
print(energies.mean(), energies.std(), energies.shape)

# %%
# Arrhenius equation k = A exp(-Ea / RT) evaluated for many temperatures at once
R = 8.314           # J/(mol K)
A = 1e13            # 1/s
Ea = 75e3           # J/mol
temperatures = np.linspace(250, 450, 9)      # 9 evenly spaced temperatures
k = A * np.exp(-Ea / (R * temperatures))
for T_, k_ in zip(temperatures, k):
    print(f"T = {T_:5.1f} K   k = {k_:9.3e} 1/s")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(1 / temperatures, np.log(k), "o-")
ax.set_xlabel("1/T (1/K)")
ax.set_ylabel("ln k")
ax.set_title("Arrhenius plot")
plt.show()

# %% [markdown]
"""
### Exercise 6.1

Plot the fraction of the acid form $f_{HA} = \frac{1}{1 + 10^{\mathrm{pH} - \mathrm{p}K_a}}$ of acetic acid
(p$K_a$ = 4.76) for pH from 0 to 14 (use `np.linspace(0, 14, 141)`). Label the axes.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
pH = np.linspace(0, 14, 141)
f_HA = 1 / (1 + 10 ** (pH - 4.76))
plt.plot(pH, f_HA)
plt.xlabel("pH"); plt.ylabel("fraction of HA")
plt.axvline(4.76, ls="--", c="gray")
plt.show()
```
</details>
"""

# %% [markdown]
"""
## 7. Tables with `pandas`

Most chemical data comes as a table: one row per molecule, one column per property. In Python, a table is a
`pandas.DataFrame`. Let's load the **ESOL** dataset: measured aqueous solubility (log S, mol/L) of 1128 compounds,
a classic benchmark for property prediction.
"""

# %%
import os
import pandas as pd

# Where does the data live? In the course repository (works both locally and on Colab).
REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chemoinformatics_tutorials/main"

def data_path(filename):
    """Local copy if available (running inside the repository), otherwise download from GitHub."""
    local = os.path.join("..", "data", filename)
    return local if os.path.exists(local) else f"{REPO_RAW}/data/{filename}"

df = pd.read_csv(data_path("esol_delaney.csv"))
df.head()          # the first 5 rows

# %%
print(df.shape)    # (rows, columns)
df.columns

# %%
# Long column names are annoying: let's rename a few
df = df.rename(columns={
    "Compound ID": "name",
    "measured log solubility in mols per litre": "logS",
    "ESOL predicted log solubility in mols per litre": "logS_ESOL",
    "Molecular Weight": "MW",
    "Number of H-Bond Donors": "HBD",
    "Number of Rings": "rings",
    "Number of Rotatable Bonds": "rot_bonds",
    "Polar Surface Area": "TPSA",
})
df.head(3)

# %%
df.describe()      # summary statistics of numeric columns

# %%
# Selecting columns
df["logS"].head()

# %%
df[["name", "smiles", "logS"]].head()

# %%
# Filtering rows with a boolean condition
soluble = df[df["logS"] > 0]
print(len(soluble), "compounds with logS > 0")
soluble[["name", "MW", "logS"]].head()

# %%
# Combine conditions with & (and), | (or); sort; take the top
small_and_soluble = df[(df["MW"] < 100) & (df["logS"] > 0)].sort_values("logS", ascending=False)
small_and_soluble[["name", "smiles", "MW", "logS"]].head(10)

# %%
# Adding a column computed from other columns
df["S_mol_per_L"] = 10 ** df["logS"]
df["is_soluble"] = df["logS"] > -2       # a boolean column
df[["name", "logS", "S_mol_per_L", "is_soluble"]].head()

# %%
# Group-by: statistics per category
df.groupby("rings")["logS"].agg(["count", "mean", "std"]).round(2)

# %%
# pandas plots directly with matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
df["logS"].hist(bins=30, ax=axes[0])
axes[0].set_xlabel("measured logS"); axes[0].set_ylabel("count")
df.plot.scatter(x="MW", y="logS", alpha=0.4, ax=axes[1])
plt.tight_layout(); plt.show()

# %%
# The ESOL model of Delaney (2004) predicted logS from 4 descriptors. How good was it?
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(df["logS"], df["logS_ESOL"], s=8, alpha=0.5)
ax.plot([-12, 2], [-12, 2], "k--", lw=1)
ax.set_xlabel("measured logS"); ax.set_ylabel("ESOL predicted logS")
r = np.corrcoef(df["logS"], df["logS_ESOL"])[0, 1]
ax.set_title(f"Pearson r = {r:.3f}")
plt.show()

# %%
# Applying a Python function to every row of a column
def solubility_class(logS):
    if logS > -1:
        return "high"
    elif logS > -3:
        return "medium"
    return "low"

df["class"] = df["logS"].apply(solubility_class)
df["class"].value_counts()

# %%
# Saving your results
df.to_csv("esol_processed.csv", index=False)
print(open("esol_processed.csv").read()[:300])

# %% [markdown]
"""
### Exercise 7.1

1. How many compounds have **no** rotatable bonds (`rot_bonds == 0`)? What is their mean `logS`?
2. Which are the 5 compounds with the **highest TPSA**? Show name, TPSA and logS.
3. Make a scatter plot of `logS` versus `TPSA`, coloured by the number of rings (`c=df["rings"]`, add `plt.colorbar()`).
   Is there a trend?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
rigid = df[df["rot_bonds"] == 0]
print(len(rigid), rigid["logS"].mean())

print(df.nlargest(5, "TPSA")[["name", "TPSA", "logS"]])

plt.scatter(df["TPSA"], df["logS"], c=df["rings"], s=10, cmap="viridis")
plt.colorbar(label="rings"); plt.xlabel("TPSA"); plt.ylabel("logS"); plt.show()
```
</details>
"""

# %% [markdown]
"""
## 8. Where to go from here

You now know enough Python to follow the rest of the course. Two habits will help you most:

1. **Read documentation and error messages.** `help(function)`, `Tab` completion, and the official docs of
   [pandas](https://pandas.pydata.org/docs/user_guide/10min.html), [numpy](https://numpy.org/doc/stable/user/absolute_beginners.html),
   [matplotlib](https://matplotlib.org/stable/tutorials/pyplot.html).
2. **Experiment.** Change a number, break a cell, fix it. Notebooks are made for that.

More practice:
- EPFL *Practical Programming in Chemistry* exercises (Schwaller group) — <https://github.com/schwallergroup/practical-programming-in-chemistry-exercises>
- MolSSI *Python scripting for computational molecular science* — <https://education.molssi.org/python_scripting_cms/>
- Kaggle's free *Python* and *Pandas* micro-courses — <https://www.kaggle.com/learn>

Next session: **01 · RDKit basics** — molecules become Python objects.
"""
