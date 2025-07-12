# Validity Checker



##### An algorithm that checks input netlist for the following parameters and flags it as valid or invalid:

##### 

##### The issues that are checked and reported will be in two parts:

* ###### Before ML -

1. Unconnected I/o - when a gate/flip flop has unconnected inputs (i.e. in high impedance Z state) - causes indeterminate output
2. Conflicting Drivers - Two sources driving the same signal - causes indeterminate output
3. Combinational loops - A drives B , B drives A without any flip flop in between 
4. Un-clocked Flip Flops

* ###### After ML-

1. Time Constraint Check , using the ML model predicted value for slack, it flags by Flag = (slack <0 ) ? Invalid : Valid


Working :
---

Implements an algorithm to check the ***Before ML*** category and if found invalid - prints that the netlist provided is invalid
due to (one or multiple issues from the category (additionally we can point out the node where the problem is encountered);
Also, in this case there is no need to run the ML computation and the delays are shown to be N.A.
---

If the circuit passes this check the ML computation runs and the ***After ML*** check is executed - i.e. if the timing constraints 
are not met it displays all predicted values and also mention that the netlist is not valid due to a negative slack.
---







---

