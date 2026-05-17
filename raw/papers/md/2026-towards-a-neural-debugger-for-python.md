---
arxiv: '2603.09951'
authors:
- Maximilian Beck
- Jonas Gehring
- Jannik Kossen
- Gabriel Synnaeve
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Towards a Neural Debugger for Python
url: https://arxiv.org/abs/2603.09951
year: 2026
---

# Towards a Neural Debugger for Python

Maximilian Beck1,∗
  
Jonas Gehring2
  
Jannik Kossen2
  
Gabriel Synnaeve2
1Johannes Kepler University Linz, Institute for Machine Learning
2Meta FAIR CodeGen Team

(March 2026)

###### Abstract

Training large language models (LLMs) on Python execution traces grounds them in code execution and enables the line-by-line execution prediction of whole Python programs, effectively turning them into *neural interpreters* (codgenteam2025\_cwm).
However, developers rarely execute programs step by step; instead, they use debuggers to stop execution at certain breakpoints and step through relevant portions only while inspecting or modifying program variables.
Existing neural interpreter approaches lack such interactive control.
To address this limitation,
we introduce *neural debuggers*: language models that emulate traditional debuggers, supporting operations such as stepping into, over, or out of functions, as well as setting breakpoints at specific source lines.
We show that neural debuggers—obtained via fine-tuning large LLMs or pre-training smaller models from scratch—can reliably model both forward execution (predicting future states and outputs) and inverse execution (inferring prior states or inputs) conditioned on debugger actions.
Evaluated on CruxEval, our models achieve strong performance on both output and input prediction tasks, demonstrating robust conditional execution modeling.
Our work takes first steps towards future agentic coding systems in which neural debuggers serve as a world model for simulated debugging environments, providing execution feedback or enabling agents to interact with real debugging tools.
This capability lays the foundation for more powerful code generation, program understanding, and automated debugging.

## 1 Introduction

Debugging is the process of isolating and correcting mistakes in computer programs (johnson1982debuggingglossary).
It is a fundamental task in software engineering that is often considered separate from writing programs (whitington2024debuggingfunctionalprogramsinterpretation).
A debugger is a collection of software tools to aid debugging (johnson1982debuggingglossary) that allows developers to inspect and control program execution through actions such as stepping into or over function calls, setting breakpoints, or returning from functions.
By observing how program states evolve line by line, developers can localize faults, understand control flow, and reason about program correctness.

Recent advances in large language models (LLMs) trained on massive code corpora have shown remarkable capabilities in code generation, completion, and repair (chen2021evaluatinglargelanguagemodels; lozhkov2024starcoder2stackv2; hui2024qwen25codertechnicalreport; guo2024deepseekcoderlargelanguagemodel; codegemmateam2024codegemmaopencodemodels; rozière2024codellamaopenfoundation).
In the coding domain, LLMs have evolved from supportive programmer tools—such as code completion—to agents writing complete codebases autonomously and are increasingly used to assist developers in debugging or fixing software bugs (handa2025economictasksperformedai).
However, open-source and academic models primarily reason over *static* code and are not explicitly grounded in program execution.

Therefore, several approaches expose language models to program execution data.
One line of work incorporates feedback such as test results (gehring2025rlef), error messages (zheng2025makeslargelanguagemodels), or runtime/shell outputs (yang2024sweagentagentcomputerinterfacesenable; wei2025swerladvancingllmreasoning) into reinforcement learning (shao2024deepseekmathpushinglimitsmathematical) or iterative generation loops (chen2023teachinglargelanguagemodels; ni2024nextteachinglargelanguage).
Another complementary direction uses *execution traces*, i.e., records of variable states and control flow transitions, to directly teach models the semantics of code execution (nye2021\_showyourwork; liu2023\_code\_exec; armengol2025\_what; codgenteam2025\_cwm) in order to improve code generation and reasoning tasks such as verification, testing, and repair.
Specifically, these approaches train language models on complete execution traces, which enable the line-by-line prediction of whole programs, effectively turning these models into *neural interpreters*.
However, this overlooks how developers actually use debuggers to interact with programs and resolve bugs.
Instead of executing programs strictly sequentially, they pause execution at certain breakpoints and step only through relevant parts while inspecting or modifying program variables.
Previous works fall short of modeling this interactive, non-sequential debugging behavior.

To address this shortcoming, we introduce *neural debuggers*: neural networks that can serve as simulated debugging environments for Python programs.
Specifically, given the program code as context, neural debuggers are language models trained to predict the line-by-line execution of a computer program, conditioned on typical debugger actions such as step\_into, step\_over, breakpoint, or step\_return.

Unlike conventional debuggers, neural debuggers can simulate program execution even for non-executable or partially specified programs, making them applicable to debugging, testing, or synthesis scenarios where access to full execution environments is unavailable or restricted (zhuo2025\_cyberzerotrainingcybersecurityagents).
Moreover, traditional debuggers require re-executing the program after code or state modifications, leading to slow iteration cycles.
In contrast, neural debuggers enable efficient reinitialization of both program state and execution context through prompting and regeneration.
Beyond standard forward execution, our neural debuggers also support approximate *inverse execution*:
given an arbitrary program state, they can infer or sample plausible preceding program states or inputs.
In agentic coding systems, neural debuggers can function as learned world models of debugging environments.
They can simulate execution feedback and external interactions—such as file reads, API requests, or operating system calls—or allow the agent to interact with real debugging environments.
Incorporating neural debuggers in this manner can augment existing coding systems with debugging and planning abilities, facilitating more effective code generation, comprehension, and repair.

In this paper, we take the first steps towards this vision by introducing neural debuggers as a Markov Decision Process (MDP), where each state represents the current program location and variable values, and transitions correspond to debugger actions that traverse a call-stack tree reconstructed from execution traces.
We introduce a data pipeline (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Towards a Neural Debugger for Python")) that supports both forward and inverse execution prediction of Python programs.
The pipeline takes execution traces as input, samples debugger trajectories, and tokenizes them into a structured format compatible with standard off-the-shelf language models.

We finetune 32 B32\text{\,}\mathrm{B}-parameter and pre-train 1.8 B1.8\text{\,}\mathrm{B}-parameter LLMs on this data to obtain neural debuggers that accurately predict program execution conditioned on debugger actions.
Our 32 B32\text{\,}\mathrm{B} parameter neural debugger consistently achieves forward next state prediction accuracies beyond 90 %90\text{\,}\mathrm{\char 37\relax} across key actions: step into, step over, step return, and breakpoint.
In addition, neural debuggers exhibit strong transfer to CruxEval output and input prediction tasks (gu2024cruxeval), demonstrating their enhanced code understanding and interpretation capabilities.
In particular, our 1.8 B1.8\text{\,}\mathrm{B}-parameter neural debugger LLM, trained from scratch on 150 B150\text{\,}\mathrm{B} tokens, attains CruxEval input and output pass@1 scores of 53.6 53.6\text{\,} and 57.7 57.7\text{\,}, respectively, while our finetuned 32 B32\text{\,}\mathrm{B}-parameter neural debugger LLM reaches scores of 66.5 66.5\text{\,} and 83.2 83.2\text{\,}, respectively.

!(/html/2603.09951/assets/x1.png)

Figure 1: Neural Debugger Data Pipeline.
Our pipeline prepares training data for neural debuggers by transforming stack-frame sequences recorded via sys.settrace in three steps: (1) we construct a state tree (Section [4.1](#S4.SS1 "4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")) from frame events; (2) we sample trajectories by traversing the state tree using a data-generating action policy; and (3) we tokenize each trajectory using our formal neural debugger language grammar (Section [4.2](#S4.SS2 "4.2 Formal language for neural debuggers ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).

In summary, in this work we make the following contributions:

* •

  We introduce neural debuggers: language models that can predict forward and inverse execution of Python programs conditioned on source code and debugger actions.
* •

  We describe a data pipeline for building neural debugger models by starting from pre-trained LLMs via finetuning or integration into pre- or mid-training data mixes.
* •

  We empirically show that neural debuggers achieve accurate intermediate state predictions and strong overall execution prediction performance.

## 2 Related work

The problem of training neural networks to simulate computer program execution, like learned interpreters, is a long standing problem and has been studied with domain-specific architectures (zaremba2014learningexecute; reed2016neuralprogrammerinterpreters; wang2020learningsemanticprogramembeddings; bieber2020\_learning\_attention\_graph) and more recently with Transformer-based LLMs. For example, nye2021\_showyourwork train Transformer models to predict intermediate states computed and source lines visited during the execution of Python functions.
They refer to this approach as “scratchpad tracing”, which they find to outperform the direct prediction of function outputs.
armengol2025\_what compare different scratchpad strategies for storing intermediate computations by training and evaluating models on different execution trace granularities, i.e., line and instruction level.
Their proposed dynamic scratchpads, in which the model updates a single self-contained scratchpad instance, produce more accurate predictions and improve performance on CruxEval output prediction tasks.
bieber2022staticpredictionruntimeerrors introduce a dataset of Python runtime errors and train a Graph Neural Network on this dataset to predict statically whether a program will encounter a runtime error when it is executed. liu2023\_code\_exec study code execution capabilities with LLMs by training small Transformer models on a large dataset of execution traces, including a curriculum of traces from programs with gradually increasing difficulty.

Collectively, these studies show that Transformer-based large language models can model control flow and variable-state dynamics during program execution, demonstrating a capability that strengthens overall code understanding.

Building on these insights, Code World Model (CWM) is the first open-weights LLM that has been trained on Python execution traces during mid-training at a large scale (codgenteam2025\_cwm).
CWM is a 32 B32\text{\,}\mathrm{B} parameter dense Transformer LLM that is trained on a large set of Python execution traces stemming from over 120 M120\text{\,}\mathrm{M} different functions, 21 k21\text{\,}\mathrm{k} executable repository images, and 262 k262\text{\,}\mathrm{k} code contest solutions.
The execution traces are formatted as sequences of observation-action pairs conditioned on the code that has been executed.
The observations correspond to serialized states of program variables, and the actions correspond to the source line being executed, while other information, such as frame event types, is encoded via special tokens.
CWM is able to reliably predict line-by-line execution of Python programs, enabling structured output prediction of programs or grounded reasoning about code generation and execution without access to live execution environments.
While it is possible to build an interactive debugger with CWM by manually steering trace prediction at inference time (codgenteam2025\_cwm, Figure B.25), it is neither possible to directly jump to future lines of code in constant time nor to predict reverse execution, function inputs, or program termination.

In this work, we introduce *neural debuggers*, which enable all of these capabilities by training language models on execution trace data, where the *next program state is conditioned on debugger actions* such as breakpoint or step into (see Figure [2](#S4.F2 "Figure 2 ‣ 4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).

## 3 Python program execution traces

Before Python code is executed by the Python interpreter, it is parsed into an abstract syntax tree and compiled into *code objects*, which contain the operations in the form of bytecode.
Code objects are created for every Python *code block*, which is executed as a unit, for example, a module, a function body, or a class definition111<https://docs.python.org/3/reference/executionmodel.html>.
The code objects are executed in Python’s *evaluation loop*, which takes the code objects and converts them into a series of stack *frame objects* (aknin2010pythoninnards; shaw2021cpython).
Each frame object contains local and global variables and the code object, which again contains the Python source line and bytecode that is being executed.
We use and record the information at Python’s frame object level to create the dataset of execution traces.

Python provides access to runtime events and these stack frame objects by setting a trace function with the signature tracefunc(frame,event,arg) via sys.settrace(tracefunc)222<https://docs.python.org/3/library/sys.html#sys.settrace>.
To collect the datasets for our neural debuggers, we use a custom tracefunc to capture execution traces containing frame, event and arg, and execute a large set of Python functions with different inputs as well as repository images with unit tests.
The frame argument is the current stack frame object, containing the local variables and the source line; the event argument is a string describing whether execution enters a new line, is about to enter or return from a function, or if an exception has occurred; and the arg argument contains event-specific data.
Python debuggers, profilers, and coverage tools use this mechanism to inspect, record, or modify a program’s state during execution.
In our case, the recorded execution traces contain the sequences of program states that serve as input data for our neural debugger pipeline, which we describe next.

## 4 Neural debugger

We introduce the concept of *neural debuggers* as neural networks that learn to simulate or predict program execution while providing the core functionalities of conventional debuggers.
Hence, a neural debugger allows for symbolic debugger-like interactions—such as stepping through execution or inspecting program states—without requiring an executable target program.
We first formalize the notion of a neural debugger by formulating the debugger as a Markov Decision Process (MDP) (Section [4.1](#S4.SS1 "4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).
Then, we introduce a structured format for training language models on debugger execution trace data (Section [4.2](#S4.SS2 "4.2 Formal language for neural debuggers ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).
Finally, we present our practical implementation of the debugger data pipeline, including our action policy for data generation, which enables large-scale training of neural debugger LLMs (Section [4.3](#S4.SS3 "4.3 Debugger trace data pipeline and dataset ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).

### 4.1 Formulating the debugger as an MDP

!(/html/2603.09951/assets/x2.png)

Figure 2: 
State-action structure of Code World Model (CWM) and neural debuggers.
In CWM (codgenteam2025\_cwm) the actions are viewed as code that modifies the variable states, while in neural debuggers actions influence the program state by controlling program execution analogous to traditional debuggers.

A debugger is an interactive software that holds the current state of the program being debugged and receives actions that control program execution, consequently determining the next state of the debugger.
Hence, we model the debugger as an interactive environment, which can be formalized as an MDP given by the tuple (\gS,\gA,P,R,s0)(\gS,\gA,P,R,s\_{0}), where
\gS\gS is the space of program states,
\gA\gA the set of available debugger actions,
P:\gS×\gA→\gSP:\gS\times\gA\rightarrow\gS the transition dynamics steered by the debugged program and its input arguments,
R:\gS×\gA→ℝR:\gS\times\gA\rightarrow\mathbb{R} is the reward function, and
s0s\_{0} the set of possible entry points for the debugger.
In this work, we do not consider a reward function, as we focus solely on state prediction.
Hence, we do not encode an action-prediction task via a reward function, which could be used to specify policies that localize specific program states quickly or mimic the debugging behavior of developers, such as not stepping through all loop iterations.

#### States.

The state contains information about the program state and runtime events recorded with sys.settrace (see Section [3](#S3 "3 Python program execution traces ‣ Towards a Neural Debugger for Python")).
Specifically, every state contains an event type (EVT), local variables and their values (LOCALS) or arguments (ARGS), and the source line of the statement being executed (SRC) (see Figure [2](#S4.F2 "Figure 2 ‣ 4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).

#### Actions.

The actions (ACTION) of neural debuggers are inspired by the interfaces of traditional debuggers such as pdb333<https://docs.python.org/3/library/pdb.html#pdbcommand-commands>.
We categorize them into *step* and *jump* actions:
step\_into (steps into a function),
step\_over (jumps over a function call or steps to the next line),
step\_return (jumps to the return statement of the current function),
breakpoint (jumps to a specified source line),
and continue (jumps to the end of the program, i.e., returns the exit code).
Since the concrete action implementations vary between debuggers, we define the outcome of the actions as transitions on the state tree, which we describe next.

#### State tree.

At runtime, computer programs maintain a call stack to store information about active subroutines, such as function calls, and to keep track of the point from where the program should continue execution after finishing the execution of a function.
In Python, the call stack is formed via references between frame objects (aknin2010pythoninnards, Interpreter Stacks).
Since the recorded program state sequences contain information about runtime events (see Section [3](#S3 "3 Python program execution traces ‣ Towards a Neural Debugger for Python")), the call stack can be reconstructed by keeping track of the order of call and return events.
Specifically, we build a tree data structure that inserts program states belonging to one function call as children of the calling line event node while still retaining the sequential order of the program states.
In this way, the depth of a node in the tree corresponds to the depth of the call stack at the given program state (see Figure [3](#S4.F3 "Figure 3 ‣ State tree. ‣ 4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).

!(/html/2603.09951/assets/x3.png)

Figure 3: 
Transition model. We visualize the state transitions as traversal on the forward and inverse state tree.
  
*Left:* Python code. *Middle:* Forward state tree with three levels indicated by indentation. *Right:* Corresponding inverse state tree. The blue numbers illustrate the correspondences between forward and inverse state tree.

#### Transitions.

Organizing execution traces as state trees provides a foundation for defining the transition model of the debugger as traversal rules on the state tree (see Figure [3](#S4.F3 "Figure 3 ‣ State tree. ‣ 4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).
Each debugger action results in a transition from a starting node to a target node on the state tree, defining how the program state evolves under debugger control:

* •

  step\_into: The target node is the immediate next node. For starting nodes with a function call, e.g., lines 1 and 3 Figure [3](#S4.F3 "Figure 3 ‣ State tree. ‣ 4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python"), traverse one level deeper in the tree; if at the end of the level, move one level up. Applying only step into actions recovers the recorded program state sequence.
* •

  step\_over: The target node is the next node at the current level. If at the end of the level, it moves one level up. It never moves a level down.
* •

  step\_return: The target node is the return node with the return event at the current level, e.g., lines 6 and 8 in Figure [3](#S4.F3 "Figure 3 ‣ State tree. ‣ 4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python").
* •

  breakpoint SRC: The target node is the first future node (at any level) that contains the specified source line. If the source line will not be visited in the future, the outcome will be the exit code, i.e., the same as for continue.
* •

  continue: Outcome is the exit code. Exit codes are normal (regular exit), error (an error or uncaught exception occurs), or never (the program enters an infinite loop).

#### Inverse program execution prediction.

Neural debuggers enable inverse program execution prediction, i.e., inferring plausible predecessor program states, inputs, or function arguments that could have produced a given program state.
This capability is particularly valuable in automated testing scenarios such as fuzzing, where diverse test inputs must be generated in a semi-random manner.
Unlike reverse debuggers, which allow backward stepping only after a forward execution has been performed and therefore traverse a fixed execution trace (engblom2012reversedebugging; savidis2021implementationlivereversedebugging), neural debuggers can start from an arbitrary program state and directly predict plausible predecessors without requiring a prior forward run.
We note that inverse prediction is inherently ambiguous: program execution is generally many-to-one, making its inverse one-to-many.
For example, reversing a sorting algorithm highlights that many distinct input orderings can yield the same sorted output.
Even a simple operation such as addition illustrates the issue: given only the sum of two variables, the original operands form an underdetermined system with infinitely many solutions.
Neural debuggers address this ambiguity by modeling and sampling from the conditional distribution over possible predecessor states, whereas traditional debuggers replay deterministic traces and typically provide neither sampling capabilities nor support for true inverse inference.

#### Inverse state tree and inverse transitions.

We construct the inverse program state tree by reversing the order of states in the forward program state tree.
In this process, we duplicate all line event nodes corresponding to function calls and assign them the event type inv\_line\_call.
The program states within the function are then attached as child nodes of these inv\_line\_call nodes, allowing the debugger to either step into or step over a function call depending on whether an inv\_step\_into or inv\_step\_over action is executed (see Figure [3](#S4.F3 "Figure 3 ‣ State tree. ‣ 4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).
Finally, breakpoint actions are disabled for inverse program prediction, and the step\_return action is repurposed as inv\_step\_call, which directly predicts a function’s input arguments.

### 4.2 Formal language for neural debuggers

We introduce a structured language format to represent the state–action sequences generated by both the forward and inverse neural debugger MDPs, designed to ensure compatibility with standard language models.
Specifically, we extend the CWM format (codgenteam2025\_cwm, Section 2.2), which includes special separator tokens that mark the beginning of state and action segments, as well as a general mechanism for serializing arbitrary Python objects into text to additionally support debugger actions and inverse execution prediction (see Section [2](#S2 "2 Related work ‣ Towards a Neural Debugger for Python") and Figure [2](#S4.F2 "Figure 2 ‣ 4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).

#### Neural debugger language grammar.

The grammar of our formal neural debugger language is shown in Figure [4](#S4.F4 "Figure 4 ‣ Neural debugger language grammar. ‣ 4.2 Formal language for neural debuggers ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python").
A neural debugger trace consists of a CODE context containing the source code under inspection, followed by a sequence of state–action pairs.
This sequence can represent either a forward or an inverse execution, beginning with an initial state and ending with an EXIT\_STATE.
The boundaries between elements are marked by special separator tokens.
Forward and inverse traces differ in their special event and action tokens, as well as in aspects of the state format.
The detailed structure of states and actions is described in Section [4.1](#S4.SS1 "4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python").

!(/html/2603.09951/assets/x4.png)

Figure 4: 
Formal neural debugger language grammar. | indicates an OR-statement, {} indicate none or more elements, and : denotes an assignment. Whitespaces are shown for illustration purposes only. <|.|> indicate special tokens, LOCALS is the local variable dictionary, ARGS are return or exception arguments, and SRC denotes the source line.

#### Local variable representation.

The textual representation of local variable dictionaries (LOCALS) must be general enough to serialize arbitrary and potentially large Python objects, while remaining compact to keep token sequence lengths manageable.
Typically, only few local variables change between lines, so we display modified variables only and insert "..":".." to indicate omitted, unchanged entries.
In the forward format, all local variables are shown after scope changes, i.e., following call, return, or breakpoint events.
During inverse prediction, the complete LOCALS dictionary is resolved at call events to ensure prediction of all input variables of the invoked function.
Following codgenteam2025\_cwm, LOCALS is serialized as JSON, and arbitrary Python objects are converted to text via their \_\_repr\_\_() methods.

### 4.3 Debugger trace data pipeline and dataset

So far, we have introduced the neural debugger MDP model and a structured text representation for state-action sequences.
Our full data pipeline produces both forward and inverse trajectories, using a stochastic policy to sample debugger actions.

#### Action policy for data generation.

To sample debugger state–action trajectories, we define a policy that selects actions conditioned on the current debugger state.
To ensure broad coverage of available actions and sufficient trajectory lengths, we employ a stochastic policy composed of mixed categorical distributions with carefully chosen probabilities (see Table [A.1](#A1.T1 "Table A.1 ‣ Action probabilities for data generation (Table A.1). ‣ A.1 Extended debugger trace dataset ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python")).
Although this setup superficially resembles behavior cloning or imitation learning (ross2011reductionimitationlearningstructured), our goal is to model state transitions rather than imitate expert behavior, as we envision coding agents to provide the actions in the future.
As shown in Figure [5](#S4.F5 "Figure 5 ‣ Dataset statistics. ‣ 4.3 Debugger trace data pipeline and dataset ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python"), this random policy produces a diverse set of states and transitions.

#### Data pipeline.

Our data pipeline takes as input a sequence of traced program states (see Section [3](#S3 "3 Python program execution traces ‣ Towards a Neural Debugger for Python")) and the source code blocks containing all source lines visited during tracing.
We then process each execution trace as follows (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Towards a Neural Debugger for Python")):
First, we build the forward or inverse state tree (see Section [4.1](#S4.SS1 "4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).
Second, we sample a debugger state-action trace trajectory from the state tree.
Third, we tokenize the trace trajectory using our structured format defined in Section [4.2](#S4.SS2 "4.2 Formal language for neural debuggers ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python").

#### Dataset statistics.

We apply our data pipeline to the function-level and repository-level execution traces from CWM, obtained by running executable Python functions and repository images (codgenteam2025\_cwm, Section 2.2).
Because repository-level traces often contain long executions with deep call stacks, we sample a single function call from the stack as the entry point for each debugger trajectory, which truncates the trace to the corresponding function scope.
Using the action policy defined in Table [A.1](#A1.T1 "Table A.1 ‣ Action probabilities for data generation (Table A.1). ‣ A.1 Extended debugger trace dataset ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python"), we obtain approximately 15 B15\text{\,}\mathrm{B} (forward + inverse) repository-level and 100 B100\text{\,}\mathrm{B} (forward + inverse) function-level debugger trajectory tokens. The stochastic nature of action and entry-point sampling effectively acts as data augmentation, providing sufficient diversity to enable multi-epoch training without overfitting.
We visualize average trajectory statistics from our function- and repository-level datasets in Figure [5](#S4.F5 "Figure 5 ‣ Dataset statistics. ‣ 4.3 Debugger trace data pipeline and dataset ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python").
Because both datasets use the same action policy, average action counts (bottom row) are similar.
However, program state event statistics and average token sequence lengths differ (see also Figure [A.1](#A1.F1 "Figure A.1 ‣ A.1 Extended debugger trace dataset ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python")): repository-level trajectories contain more function calls (i.e., call and return events), more exceptions, and generally longer token sequences for the same number of actions.
This is primarily due to larger local variable dictionaries and the presence of more arbitrary Python objects in repository-level executions.
The following experiments use a mixture of function- and repository-level trajectories in both forward and inverse directions.

!(/html/2603.09951/assets/x5.png)

Figure 5: 
Average token, action and event counts of forward debugger trajectory datasets. We show the mean function-level counts in turquoise and the repository-level counts in yellow, with the boxes indicating the 25% and 90% range. While the average action counts are similar due to the same action policy, repository-level trajectories contain more function calls, more exceptions, and longer token sequences.

## 5 Experimental results

In this section, we explore the feasibility of training neural debuggers for forward and inverse debugging through a systematic empirical evaluation.
Our experiments aim to answer the following questions:
(i) How does finetuning large models that have already been pre-trained on trace data with a different format (e.g., CWM (codgenteam2025\_cwm)) compare to pre-training smaller Transformer models from scratch (Section [5.1](#S5.SS1 "5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python"))?
(ii) What are the prediction accuracies of individual actions’ state elements(Section [5.1](#S5.SS1 "5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and [5.2](#S5.SS2 "5.2 Next program state prediction by state component ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python"))?
(iii) How well do neural debuggers perform on related downstream tasks such as input and output prediction, and how sensitive is this performance to the prediction horizon (Section [5.3](#S5.SS3 "5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python"))?

#### Experiment setup.

We train neural debugger models by finetuning and pre-training decoder-only Transformer language models with different data mixes and evaluate their forward and inverse program execution prediction capabilities.
Our neural debugger dataset consists of equal proportions of function-level and repository-level execution traces, covering both forward and inverse directions.
We finetune the 32 B32\text{\,}\mathrm{B}-parameter CWM model for 50 B50\text{\,}\mathrm{B} tokens using a linear warmup followed by a constant learning rate, training exclusively on debugger trace data.
In addition, we pre-train smaller 1.8 B1.8\text{\,}\mathrm{B}-parameter Transformer models on 50 B50\text{\,}\mathrm{B} and 150 B150\text{\,}\mathrm{B} tokens using a linear warmup and a cosine learning rate decay schedule, and explore three data mixtures:
debugger trace data only and two different data mixtures of debugger trace data with web data from DCLM (li2025datacomplmsearchgenerationtraining) and GitHub code data.
Further training details are provided in Appendix [B.1](#A2.SS1 "B.1 Training recipe ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python").

### 5.1 Finetuning and pre-training neural debuggers

!(/html/2603.09951/assets/x6.png)

(a)

!(/html/2603.09951/assets/x7.png)

(b)

Figure 6: Evolution of the exact match (em) next state prediction accuracy per debugger action on the function-level validation set during training. (a) Forward execution prediction. (b) Inverse execution prediction. Figure [B.1](#A2.F1 "Figure B.1 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python") shows the corresponding results for the repository-level validation set, which lead to the same conclusions.

To compare finetuning and pre-training approaches, we evaluate next-state prediction accuracy per debugger action throughout training.
For each action, we sample 800 trajectories from the validation set using our debugger data pipeline (Section [4.3](#S4.SS3 "4.3 Debugger trace data pipeline and dataset ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")) and truncate each trace after the respective action to form the evaluation prompt.

In the following, we discuss next-state prediction performance presented in Figure [6](#S5.F6 "Figure 6 ‣ 5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") for function-level, and in Figure [B.1](#A2.F1 "Figure B.1 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python") for repository-level data.
We report exact-match accuracy between model predictions (with greedy decoding) and ground-truth program states.
Our results for function-level and repository-level evaluations lead to the same conclusions, which is why we place Figure [B.1](#A2.F1 "Figure B.1 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python") in the appendix.

#### Step actions are easier than jump actions.

By comparing the left two columns to the right two columns in Figure [6(a)](#S5.F6.sf1 "Figure 6(a) ‣ Figure 6 ‣ 5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and [1(a)](#A2.F1.sf1 "Figure 1(a) ‣ Figure B.1 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python"), we observe that step actions (e.g., step into, step over) achieve higher exact-match accuracy than jump actions (e.g., step return, breakpoint).
While step actions plateau early, jump actions continue to improve with more training tokens.
This difference arises because predicting transitions to the next source line is simpler than predicting multi-line transitions, and single-line transitions occur far more frequently in the dataset (Figure [5](#S4.F5 "Figure 5 ‣ Dataset statistics. ‣ 4.3 Debugger trace data pipeline and dataset ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).
In addition, in Figure [6](#S5.F6 "Figure 6 ‣ 5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and [B.1](#A2.F1 "Figure B.1 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python") the finetuned CWM model shows a rapid performance increase over the first few tokens, indicating a benefit from its pre-training and prior exposure to trace data during its mid-training phase, even though the trace data format differed and did not include debugger actions (codgenteam2025\_cwm).

#### Inverse execution is learnable.

In Figure [6(b)](#S5.F6.sf2 "Figure 6(b) ‣ Figure 6 ‣ 5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and [1(b)](#A2.F1.sf2 "Figure 1(b) ‣ Figure B.1 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python"), we observe a similar trend for inverse execution prediction as for forward execution prediction in Figure [6(a)](#S5.F6.sf1 "Figure 6(a) ‣ Figure 6 ‣ 5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and [1(a)](#A2.F1.sf1 "Figure 1(a) ‣ Figure B.1 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python").
Even though inverse prediction accuracies are lower than forward accuracies, they consistently improve over training.
We note that for the inv\_step\_call action in the right column of Figure [6(b)](#S5.F6.sf2 "Figure 6(b) ‣ Figure 6 ‣ 5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and [1(b)](#A2.F1.sf2 "Figure 1(b) ‣ Figure B.1 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python"), only the trend of the exact match @1 metric is indicative, not the absolute numbers, since the exact match @1 metric does not capture the inherent ambiguity in the inverse program states.
Since forward prediction is deterministic (in most cases), perfect accuracy is achievable. However, for inverse prediction, the inherent ambiguity provides a data-dependent upper bound on performance.
In contrast, in Table [1](#S5.T1 "Table 1 ‣ Neural debuggers excel at output prediction. ‣ 5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python"), we compute the pass@1 metric with assert f(predicted\_input) == reference\_output; i.e., accounting for the ambiguity, we observe that input prediction accuracies are comparable to the output prediction.
We refer to Appendix [A.2](#A1.SS2 "A.2 Evaluating Inverse Execution Prediction ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python") for details on the evaluation of inverse execution prediction.

#### Small models are good neural debuggers.

Comparing the finetuned 32 B32\text{\,}\mathrm{B}-parameter CWM model with our 1.8 B1.8\text{\,}\mathrm{B}-parameter models trained on the same 50 B50\text{\,}\mathrm{B} tokens in Figure [6(a)](#S5.F6.sf1 "Figure 6(a) ‣ Figure 6 ‣ 5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and [1(a)](#A2.F1.sf1 "Figure 1(a) ‣ Figure B.1 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python"),
we observe a modest ∼\sim5 %5\text{\,}\mathrm{\char 37\relax} point gap for step actions and a larger >>15 %15\text{\,}\mathrm{\char 37\relax} point gap for more complicated jump actions.
However, extending pre-training to 150 B150\text{\,}\mathrm{B} tokens substantially narrows both gaps, suggesting that small Transformers can already serve as capable neural debuggers.
Furthermore, experiments with different data mixtures indicate that debugger data can be integrated into existing pre- or mid-training corpora, similar to CWM’s mid-training strategy (codgenteam2025\_cwm).

### 5.2 Next program state prediction by state component

For more fine-grained performance analysis, we evaluate next-state prediction accuracies by debugger action and state element (Figure [2](#S4.F2 "Figure 2 ‣ 4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).
Here, we train all models exclusively on debugger trace tokens and report accuracies for four components: local variable dictionaries, return or exception arguments, source lines, and state events.

#### Source lines & state events are predicted reliably; local variables contain errors.

Across all actions and datasets, our models consistently achieve high accuracies for source line and state event prediction in both forward and inverse modes for function-level (see Figure [7](#S5.F7 "Figure 7 ‣ Source lines & state events are predicted reliably; local variables contain errors. ‣ 5.2 Next program state prediction by state component ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python")) and repository-level (see Figure [B.2](#A2.F2 "Figure B.2 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python")) data.
In addition, we find that predicting source lines for function-level data achieves higher exact match scores than for repository-level data, especially for step\_return action prediction of the 1.8 B1.8\text{\,}\mathrm{B} models, which exhibits a ∼\sim5 %5\text{\,}\mathrm{\char 37\relax} point (function-level) and ∼\sim10 %10\text{\,}\mathrm{\char 37\relax} point (repository-level) drop in source line accuracy (see em\_src@1 in Figure [7(a)](#S5.F7.sf1 "Figure 7(a) ‣ Figure 7 ‣ Source lines & state events are predicted reliably; local variables contain errors. ‣ 5.2 Next program state prediction by state component ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and [2(a)](#A2.F2.sf1 "Figure 2(a) ‣ Figure B.2 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python")).
We hypothesize that this is due to more complex conditional branching in repository-level data.
Most remaining errors stem from local variable and return/exception argument predictions, which show markedly lower accuracies than source line and state event predictions.
We further find that the accuracy gaps between the finetuned CWM model and smaller pre-trained models—particularly for jump actions such as breakpoint or step\_return—originate primarily from these local variables and argument components (e.g., em\_locals, em\_arg) instead of source line and event components (e.g., em\_src, em\_evt).
This indicates that predicting variables values is harder than predicting control flow, especially for smaller models.
Note that, similar to the previous section, em\_locals@1 does for inv\_step\_call actions in Figure [7(b)](#S5.F7.sf2 "Figure 7(b) ‣ Figure 7 ‣ Source lines & state events are predicted reliably; local variables contain errors. ‣ 5.2 Next program state prediction by state component ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and [2(b)](#A2.F2.sf2 "Figure 2(b) ‣ Figure B.2 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python"), it does not account for ambiguities in inverse execution prediction (see Section [A.2](#A1.SS2 "A.2 Evaluating Inverse Execution Prediction ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python")).
Entries marked “N/A” indicate state components not present in a given trace element (see Figure [4](#S4.F4 "Figure 4 ‣ Neural debugger language grammar. ‣ 4.2 Formal language for neural debuggers ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).

!(/html/2603.09951/assets/x8.png)

(a)

!()

(b)

Figure 7: Exact match (em) next state prediction by state component per debugger action on the function-level validation set. Most prediction errors are in the local variables and return or exception arguments, while source lines and state events are predicted reliably. Figure [B.2](#A2.F2 "Figure B.2 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python") shows the corresponding results for the repository-level data. Conclusions are the same.

### 5.3 Input and output prediction on CruxEval

To compare the code understanding and execution capabilities of neural debuggers with general-purpose language models, we evaluate them on the CruxEval input–output prediction benchmark (gu2024cruxeval).
We generate debugger execution traces from CruxEval’s Python functions and define prediction tasks corresponding to debugger jump actions: step\_return and breakpoint for output prediction (Figure [B.3](#A2.F3 "Figure B.3 ‣ B.3 CruxEval input and output prediction prompts in neural debugger format ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python")), and inv\_step\_call for input prediction (Figure [B.4](#A2.F4 "Figure B.4 ‣ B.3 CruxEval input and output prediction prompts in neural debugger format ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python")).
For each case, we construct custom prompts such that the target state after the respective jump action contains either the output value or the input arguments of the function.

#### Neural debuggers excel at output prediction.

Table [1](#S5.T1 "Table 1 ‣ Neural debuggers excel at output prediction. ‣ 5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") summarizes CruxEval input and output prediction pass@1 scores calculated with assert f(input) == output for our models.
We observe consistently high input and output prediction performance, with breakpoint outperforming step\_return for output prediction.
We attribute this to the prompt design: the breakpoint action explicitly includes the source line associated with the return statement, helping the model localize the correct execution context and focus on predicting the function’s output rather than identifying the relevant line.
Our neural debugger model finetuned from CWM achieves a CruxEval score of 83.2 %83.2\text{\,}\mathrm{\char 37\relax} using the breakpoint action.
Even without providing the return line in the prompt with the step\_return action, our neural debugger model finetuned from CWM achieves a CruxEval score of 77.9 %77.9\text{\,}\mathrm{\char 37\relax}, corresponding to a 19.8 %19.8\text{\,}\mathrm{\char 37\relax}-point improvement compared to the stock CWM model evaluated with the CWM execution trace format (codgenteam2025\_cwm, Table 8, Trace Step, 58.1 %58.1\text{\,}\mathrm{\char 37\relax}).
Already, the smaller 1.8 B1.8\text{\,}\mathrm{B} Transformer trained from scratch achieves 57.7 %57.7\text{\,}\mathrm{\char 37\relax} and 48.0 %48.0\text{\,}\mathrm{\char 37\relax} with breakpoint and step\_return action on the same task after 150 B150\text{\,}\mathrm{B} training tokens, highlighting that training exclusively on debugger trace data confers strong execution reasoning abilities.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Tokens | Input | Output | |
| inv\_step\_call | step\_return | breakpoint |
| 1.8B Transformer Pretrain | 50B | 40.7 | 34.4 | 44.9 |
| 1.8B Transformer Pretrain | 150 B150\text{\,}\mathrm{B} | 53.6 | 48.0 | 57.7 |
| 32B CWM Finetune | 50B | 66.5 | 77.9 | 83.2 |

Table 1: CruxEval input and output pass@1 scores for single step prediction with neural debugger actions inv\_step\_call, step\_return, and breakpoint (greedy decoding). All models are trained on debugger trace data only.

#### Prediction accuracy decreases with prediction horizon.

We next analyze how input and output prediction accuracy varies with the prediction horizon.
For each CruxEval Python function, we generate multiple prompts by inserting a varying number of step\_into, step\_over, or inv\_step\_over actions before the final jump action (breakpoint, step\_return, or inv\_step\_over).
This setup forces the final jump action to skip a variable number of intermediate program states, ranging from the next state (single-step prediction) to direct prediction of the full function’s input or output.
We normalize the number of skipped states by the total number of states in the function and group examples with similar normalized horizons into bins.
Figure [8](#S5.F8 "Figure 8 ‣ Prediction accuracy decreases with prediction horizon. ‣ 5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") plots the fraction of skipped states on the x-axis (0 corresponding to single-step prediction and 1 corresponding to full input/output prediction as in Table [1](#S5.T1 "Table 1 ‣ Neural debuggers excel at output prediction. ‣ 5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python")), and plots input and output prediction accuracy on the y-axis for different exact match @k thresholds (analogous to pass@k, except for input prediction in Figure [8(a)](#S5.F8.sf1 "Figure 8(a) ‣ Figure 8 ‣ Prediction accuracy decreases with prediction horizon. ‣ 5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python"); see Section [A.2](#A1.SS2 "A.2 Evaluating Inverse Execution Prediction ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python") for details).
For both models, the finetuned 32 B32\text{\,}\mathrm{B} CWM neural debugger and the 1.8 B1.8\text{\,}\mathrm{B} model trained from scratch, accuracy decreases as the prediction horizon increases, with a steeper decline for smaller models trained from scratch.
Larger sampling budgets (higher kk) partially mitigate the accuracy drop, suggesting that test-time ensembling, majority voting, or resampling based on model uncertainty could serve as effective strategies to further improve neural debugger performance.
We believe this result offers exciting avenues for future research: for example, one could investigate whether model uncertainties are suitable for deciding when to jump to return events, adaptively assigning inference compute based on program difficulty.

!(/html/2603.09951/assets/x10.png)

(a)

!(/html/2603.09951/assets/x11.png)

(b)

!(/html/2603.09951/assets/x12.png)

(c)

Figure 8: CruxEval input and output exact match @k scores for increasing prediction horizon (see Section [A.2](#A1.SS2 "A.2 Evaluating Inverse Execution Prediction ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python") for a discussion on exact match vs. pass metric). We use the same prompts (see Section [B.3](#A2.SS3 "B.3 CruxEval input and output prediction prompts in neural debugger format ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python")) as for Table [1](#S5.T1 "Table 1 ‣ Neural debuggers excel at output prediction. ‣ 5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and insert step actions to increase number of revealed program states in the prompt. Long prediction horizons correspond to a high fraction of skipped frames (close to 1) and short prediction horizons to a low fraction (close to 0). We generate responses with temperature 0.6 and top-p 0.95. Across all models, the prediction accuracy decreases as the prediction horizon increases,
with a steeper decline for smaller models trained from scratch..

## 6 Limitations and future work

Neural debuggers represent a step towards program execution-aware language models with promising practical applications.
At this stage, however, several limitations remain that suggest clear directions for further research.

#### Agentic program repair, reasoning & tool use with neural debuggers.

As a first downstream application, we evaluated neural debuggers on input and output prediction.
We believe that including neural debuggers in agentic coding and extending them to tasks such as program repair and bug fixing offer particularly promising opportunities.
Such applications could benefit from LLMs that self-debug their generated code during reasoning or from controlling a debugger in real debugging environments.

#### Expanding & improving data generation.

Thus far, we have applied neural debuggers exclusively to Python programs and used random action policies to generate debugger trajectories.
Future work could expand the dataset with execution traces from additional programming languages and develop more structured or goal-directed action policies.
For example, incorporating syntactic code information—such as compound statements444<https://docs.python.org/3/reference/compound_stmts.html>, e.g., conditions or loops—into the data generating policy could further bias trajectories toward semantically richer transitions and improve data quality.
As a result, we would expect different action distributions for different code data sources (e.g., for function-level and repository-level data in Figure [5](#S4.F5 "Figure 5 ‣ Dataset statistics. ‣ 4.3 Debugger trace data pipeline and dataset ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")).

#### Improving inverse debugging.

Further research on neural debuggers could target improvements in local variable prediction through better modeling of ambiguity and feasible value sets.
Additionally, evaluation metrics for inverse prediction should be adapted to account for multiple valid traces or outcomes.
Such metrics could capture more nuanced prediction errors of neural debuggers, especially for inverse debugging, which would naturally extend our analysis in Section [5.2](#S5.SS2 "5.2 Next program state prediction by state component ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and [5.3](#S5.SS3 "5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python").

#### Better Python object representations.

Our current approach serializes arbitrary Python objects to text using built-in mechanisms.
While this simplifies data collection, it becomes infeasible for very large or complex data structures.
We observed this in the longer state-action trajectories for repository-level data (see Figure [A.1](#A1.F1 "Figure A.1 ‣ A.1 Extended debugger trace dataset ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python")).
Developing compact neural representations of arbitrary Python objects remains an interesting problem for future research.

## 7 Conclusion

In this work, we introduced the concept of *neural debuggers*—neural networks capable of predicting the line-by-line execution of computer programs conditioned on common debugger actions such as step\_into, step\_over, breakpoint, and step\_return.
We formalized the neural debugger as a Markov Decision Process (MDP), where states comprise program variables and source lines, and transitions are defined by traversal rules on a tree structure reconstructed from execution traces via the program’s call stack.
Our experiments show that finetuning existing large language models or pre-/mid-training on debugger trace data yields neural debuggers that achieve accurate intermediate state predictions and strong overall execution prediction performance.
We believe that neural debuggers represent a promising step toward language models that are explicitly grounded in program execution.
By learning to model step-wise execution dynamics and debugger control flow within a neural framework, they open a path toward integrating reasoning and execution within a single learned system.
Looking forward, we envision neural debuggers as a core component of future agentic coding systems, acting as a world model for simulated debugging environments and enabling agents to interact with real debuggers through execution-aware feedback.
In this way, neural debuggers tightly couple neural reasoning with executable program behavior and have the potential to substantially advance code generation, understanding, and debugging.

## References

\beginappendix

## Appendix A Extended neural debugger

### A.1 Extended debugger trace dataset

!(/html/2603.09951/assets/x13.png)

Figure A.1: 
Length distributions of the debugger trace dataset. We show the length distribution histograms of debugger trajectories in number of tokens and actions, and show the distribution of the code length in number of characters.
Even though the number of actions are similar for function-level and repository-level data, the average token count for repository-level trajectories is significantly higher (see also Section [4.3](#S4.SS3 "4.3 Debugger trace data pipeline and dataset ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")). The reasons for the longer trajectories are more arbitrary python objects in their local variable state dictionaries and larger code context (see right column).

#### Action probabilities for data generation (Table [A.1](#A1.T1 "Table A.1 ‣ Action probabilities for data generation (Table A.1). ‣ A.1 Extended debugger trace dataset ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python")).

We generate our debugger trace trajectory dataset by sampling actions from a mixture of two categorical distributions, each selected with equal probability (see Table [A.1](#A1.T1 "Table A.1 ‣ Action probabilities for data generation (Table A.1). ‣ A.1 Extended debugger trace dataset ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python")).
The first distribution samples uniformly from the full set of available actions to ensure that every action type is represented in the trajectories.
However, actions such as breakpoint, step return, or continue can skip over large portions of the program’s execution, resulting in shorter trajectories.
To mitigate this effect, we introduce a second distribution that samples exclusively from the step into and step over actions, with equal probability between them.
This balanced mixture encourages sufficient trajectory length while maintaining diversity in the action space.

| Policy Prob | step\_into | step\_over | step\_return | breakpoint | continue |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.35 | 0.1 | 0.2 | 0.1 | 0.05 |
| 0.5 | 0.5 | 0.5 | - | - | - |

Table A.1: Action probabilities for the action policy mix used to generate our debugger trace dataset.

### A.2 Evaluating Inverse Execution Prediction

The standard way to evaluate input and output predictions is to execute the code and check its correctness with assert statements and given reference inputs and outputs (gu2024cruxeval).
For input prediction, we use assert(predicted\_input) == reference\_output, while for output prediction, we use assert(reference\_input) == predicted\_output.
In order to evaluate our neural debuggers on CruxEval, we collect ground truth execution traces for all examples with the provided reference inputs.
We then use these traces to create prompts and targets for the evaluations in Table [1](#S5.T1 "Table 1 ‣ Neural debuggers excel at output prediction. ‣ 5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and Figure [8](#S5.F8 "Figure 8 ‣ Prediction accuracy decreases with prediction horizon. ‣ 5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python").
Similarly, we use the recorded execution traces (see Section [4.3](#S4.SS3 "4.3 Debugger trace data pipeline and dataset ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")) from our validation sets to generate prompts and targets from these execution trace validation sets for Figure [6](#S5.F6 "Figure 6 ‣ 5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and Figure [7](#S5.F7 "Figure 7 ‣ Source lines & state events are predicted reliably; local variables contain errors. ‣ 5.2 Next program state prediction by state component ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python").
To compute the metrics, we first parse the predictions and then compare the individual state components (see Figure [2](#S4.F2 "Figure 2 ‣ 4.1 Formulating the debugger as an MDP ‣ 4 Neural debugger ‣ Towards a Neural Debugger for Python")) to the ground truth via exact match.

For forward debugger trace prediction, this approach corresponds to the metric computation in CruxEval, as most functions and repositories in our validation sets and CruxEval functions are deterministic.
In fact, CruxEval even applies a filter to contain only deterministic functions555<https://github.com/facebookresearch/cruxeval/blob/main/data/README.md>.
In this case, exact\_match@1 and pass@1 metric scores are identical.

However, programs that are deterministic in forward execution must not be deterministic in inverse execution; e.g., remember our previous example of the inverse execution of a sum operation.
Similarly, our example in Figure [C.1](#A3.F1 "Figure C.1 ‣ Appendix C Neural debugger trace example ‣ Towards a Neural Debugger for Python") illustrates this fact.
Therefore, using only a single reference input candidate to compute exact match scores for predicted function arguments can lead to an underestimation of performance, as we show in Table [A.2](#A1.T2 "Table A.2 ‣ A.2 Evaluating Inverse Execution Prediction ‣ Appendix A Extended neural debugger ‣ Towards a Neural Debugger for Python").
For the interpretation of our results, this means that our exact match scores still provide a signal about the relative performance between models, but they may underestimate the true absolute performance.
To address this problem, we plan to develop dedicated evaluations and benchmarks focused on neural debugger capabilities, using executable functions and/or Docker images to produce target traces based on data (such as inputs or actions) generated by the model.

|  |  |  |  |
| --- | --- | --- | --- |
| Model | Tokens | Input | Input |
| inv\_step\_call | inv\_step\_call |
|  |  | exact\_match@1 | pass@1 |
| 1.8B Transformer Pretrain | 50B | 14.3 | 40.7 |
| 1.8B Transformer Pretrain | 150 B150\text{\,}\mathrm{B} | 17.7 | 53.6 |
| 32B CWM Finetune | 50B | 23.1 | 66.5 |

Table A.2: Comparison of CruxEval input exact\_match@1 and pass@1 scores for single step prediction with the neural debugger action inv\_step\_call (greedy decoding) corresponding to Table [1](#S5.T1 "Table 1 ‣ Neural debuggers excel at output prediction. ‣ 5.3 Input and output prediction on CruxEval ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python"). The exact\_match@1 (em@1) scores are considerably lower than pass@1 score, since exact match only compares to the reference inputs, and does not account for potential ambiguities in input predictions.

## Appendix B Extended experiments

### B.1 Training recipe

We train our neural debuggers using the AdamW optimizer (loshchilov2018decoupled) with weight decay 0.1 0.1\text{\,}.
We use the Llama-2 architecture for the pre-training of our neural debugger models with 1.8 B1.8\text{\,}\mathrm{B} parameters (touvron2023llama2openfoundation).
For pre-training we use a learning rate schedule consisting of 750 750\text{\,} linear warmup steps to the peak learning rate of 1×10−3 1\text{\times}{10}^{-3}\text{\,}, followed by a cosine decay to learning rate zero, over the remaining training steps.
In contrast, for finetuning, we use a linear warmup over 750 750\text{\,} steps, followed by a constant peak learning rate of 1×10−5 1\text{\times}{10}^{-5}\text{\,}.
For all experiments we use a sequence length of 16 384 16\,384\text{\,} and a batch size of 1 M1\text{\,}\mathrm{M} tokens or 64 64\text{\,} sequences.

### B.2 Extended next state prediction results

In the main text in Section [5.1](#S5.SS1 "5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") and Section [5.2](#S5.SS2 "5.2 Next program state prediction by state component ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python"), we show the results for the function-level data.
In addition, we report the analogous results for the repository-level data.

In Figure [B.1](#A2.F1 "Figure B.1 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python"), we show the evolution of the next state prediction accuracy per debugger action on the repository-level validation set during training.

!(/html/2603.09951/assets/x14.png)

(a)

!(/html/2603.09951/assets/x15.png)

(b)

Figure B.1: Evolution of the exact match (em) next state prediction accuracy per debugger action on the repository-level validation set during training. (a) Forward execution prediction. (b) Inverse execution prediction. Figure [6](#S5.F6 "Figure 6 ‣ 5.1 Finetuning and pre-training neural debuggers ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") shows the corresponding results for the function-level validation set. The results on both validation sets lead to similar conclusions.

In Figure [B.2](#A2.F2 "Figure B.2 ‣ B.2 Extended next state prediction results ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python"), we show the next state prediction results by state component per debugger action on the repository-level validation set.

!(/html/2603.09951/assets/x16.png)

(a)

!(/html/2603.09951/assets/x17.png)

(b)

Figure B.2: Exact match (em) next state prediction by state component per debugger action on the repository-level validation set. Figure [7](#S5.F7 "Figure 7 ‣ Source lines & state events are predicted reliably; local variables contain errors. ‣ 5.2 Next program state prediction by state component ‣ 5 Experimental results ‣ Towards a Neural Debugger for Python") shows the corresponding results for the function-level validation set. Conclusions remain unchanged.

### B.3 CruxEval input and output prediction prompts in neural debugger format

In this section, we show examples of how we evaluate our neural debuggers on the CruxEval input and output prediction tasks.

In Figure [B.3](#A2.F3 "Figure B.3 ‣ B.3 CruxEval input and output prediction prompts in neural debugger format ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python"), we show our prompts for CruxEval output prediction with the step\_return and breakpoint action.

[⬇](data:text/plain;base64,PHxiZWdpbl9vZl90ZXh0fD48fHRyYWNlX2NvbnRleHRfc3RhcnR8PgpkZWYgZihzaW5nbGVfZGlnaXQpOgogICAgcmVzdWx0ID0gW10KICAgIGZvciBjIGluIHJhbmdlKDEsIDExKToKICAgICAgICBpZiBjICE9IHNpbmdsZV9kaWdpdDoKICAgICAgICAgICAgcmVzdWx0LmFwcGVuZChjKQogICAgcmV0dXJuIHJlc3VsdAoKZGVmIG1haW4oKToKICAgIHJldHVybiBmKDUpCgo8fGZyYW1lX3NlcHw+PHxjYWxsX3NlcHw+Cjx8c3JjX3NlcHw+ZGVmIG1haW4oKToKPHxhcmdfc2VwfD57fQo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGxpbmVfc2VwfD4KPHxhcmdfc2VwfD57fQo8fHNyY19zZXB8PiAgICByZXR1cm4gZig1KQo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGNhbGxfc2VwfD4KPHxzcmNfc2VwfD5kZWYgZihzaW5nbGVfZGlnaXQpOgo8fGFyZ19zZXB8Pnsic2luZ2xlX2RpZ2l0IjogIjUifQooKkBcdGV4dGNvbG9ye2N3bV9zdHJpbmdfY29sb3JfYnJpZ2h0fXs8fGFjdGlvblxfc2VwfD48fHN0ZXBcX3JldHVybnw+fUAqKQoKPHxmcmFtZV9zZXB8PgotLS0tRU5EIE9GIFBST01QVC0tLS0KCgo8fHJldHVybl9zZXB8Pgo8fHNyY19zZXB8PiAgICByZXR1cm4gcmVzdWx0Cjx8YXJnX3NlcHw+IlsxLCAyLCAzLCA0LCA2LCA3LCA4LCA5LCAxMF0i)

<|begin\_of\_text|><|trace\_context\_start|>

def f(single\_digit):

result = []

for c in range(1, 11):

if c != single\_digit:

result.append(c)

return result

def main():

return f(5)

<|frame\_sep|><|call\_sep|>

<|src\_sep|>def main():

<|arg\_sep|>{}

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{}

<|src\_sep|> return f(5)

<|action\_sep|><|step\_into|>

<|frame\_sep|><|call\_sep|>

<|src\_sep|>def f(single\_digit):

<|arg\_sep|>{"single\_digit": "5"}

<|action\_sep|><|step\_return|>

<|frame\_sep|>

----END OF PROMPT----

<|return\_sep|>

<|src\_sep|> return result

<|arg\_sep|>"[1, 2, 3, 4, 6, 7, 8, 9, 10]"

(a) Output prediction with step\_return action.

[⬇](data:text/plain;base64,PHxiZWdpbl9vZl90ZXh0fD48fHRyYWNlX2NvbnRleHRfc3RhcnR8PgpkZWYgZihzaW5nbGVfZGlnaXQpOgogICAgcmVzdWx0ID0gW10KICAgIGZvciBjIGluIHJhbmdlKDEsIDExKToKICAgICAgICBpZiBjICE9IHNpbmdsZV9kaWdpdDoKICAgICAgICAgICAgcmVzdWx0LmFwcGVuZChjKQogICAgcmV0dXJuIHJlc3VsdAoKZGVmIG1haW4oKToKICAgIHJldHVybiBmKDUpCgo8fGZyYW1lX3NlcHw+PHxjYWxsX3NlcHw+Cjx8c3JjX3NlcHw+ZGVmIG1haW4oKToKPHxhcmdfc2VwfD57fQo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGxpbmVfc2VwfD4KPHxhcmdfc2VwfD57fQo8fHNyY19zZXB8PiAgICByZXR1cm4gZig1KQo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGNhbGxfc2VwfD4KPHxzcmNfc2VwfD5kZWYgZihzaW5nbGVfZGlnaXQpOgo8fGFyZ19zZXB8Pnsic2luZ2xlX2RpZ2l0IjogIjUifQooKkBcdGV4dGNvbG9ye2N3bV9zdHJpbmdfY29sb3JfYnJpZ2h0fXs8fGFjdGlvblxfc2VwfD48fGJyZWFrcG9pbnR8PiAgICByZXR1cm4gcmVzdWx0fUAqKQoKPHxmcmFtZV9zZXB8PgotLS0tRU5EIE9GIFBST01QVC0tLS0KCjx8bGluZV9zZXB8Pgo8fGFyZ19zZXB8Pnsic2luZ2xlX2RpZ2l0IjogIjUiLAoicmVzdWx0IjogIlsxLCAyLCAzLCA0LCA2LCA3LCA4LCA5LCAxMF0iLCAiYyI6ICIxMCJ9Cjx8c3JjX3NlcHw+ICAgIHJldHVybiByZXN1bHQ=)

<|begin\_of\_text|><|trace\_context\_start|>

def f(single\_digit):

result = []

for c in range(1, 11):

if c != single\_digit:

result.append(c)

return result

def main():

return f(5)

<|frame\_sep|><|call\_sep|>

<|src\_sep|>def main():

<|arg\_sep|>{}

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{}

<|src\_sep|> return f(5)

<|action\_sep|><|step\_into|>

<|frame\_sep|><|call\_sep|>

<|src\_sep|>def f(single\_digit):

<|arg\_sep|>{"single\_digit": "5"}

<|action\_sep|><|breakpoint|> return result

<|frame\_sep|>

----END OF PROMPT----

<|line\_sep|>

<|arg\_sep|>{"single\_digit": "5",

"result": "[1, 2, 3, 4, 6, 7, 8, 9, 10]", "c": "10"}

<|src\_sep|> return result

(b) Output prediction with breakpoint action.

Figure B.3: CruxEval output prompt for the step\_return and the breakpoint actions.
The outcome of the step\_return action is the frame with the return event and argument, while the outcome of the breakpoint action is the line event frame, including the local variables dictionary at the source line of the return statement.
Predictions are generated by the 1.8 B1.8\text{\,}\mathrm{B} parameter neural debugger trained on 150 B150\text{\,}\mathrm{B} debugger trace tokens. Some line breaks are inserted for illustration purposes.

In Figure [B.4](#A2.F4 "Figure B.4 ‣ B.3 CruxEval input and output prediction prompts in neural debugger format ‣ Appendix B Extended experiments ‣ Towards a Neural Debugger for Python"), we show the prompt for CruxEval input prediction with the inv\_step\_call action.

[⬇](data:text/plain;base64,PHxiZWdpbl9vZl90ZXh0fD48fHRyYWNlX2NvbnRleHRfc3RhcnR8PgpkZWYgZihzaW5nbGVfZGlnaXQpOgogICAgcmVzdWx0ID0gW10KICAgIGZvciBjIGluIHJhbmdlKDEsIDExKToKICAgICAgICBpZiBjICE9IHNpbmdsZV9kaWdpdDoKICAgICAgICAgICAgcmVzdWx0LmFwcGVuZChjKQogICAgcmV0dXJuIHJlc3VsdAoKPHxmcmFtZV9zZXB8Pjx8aW52X3JldHVybl9zZXB8Pgo8fHNyY19zZXB8PiAgICByZXR1cm4gcmVzdWx0Cjx8YXJnX3NlcHw+IlsxLCAyLCAzLCA0LCA2LCA3LCA4LCA5LCAxMF0iCigqQFx0ZXh0Y29sb3J7Y3dtX3N0cmluZ19jb2xvcl9icmlnaHR9ezx8YWN0aW9uXF9zZXB8Pjx8aW52XF9zdGVwXF9jYWxsfD59QCopCgo8fGZyYW1lX3NlcHw+Ci0tLS1FTkQgT0YgUFJPTVBULS0tLQoKPHxpbnZfY2FsbF9zZXB8Pgo8fHNyY19zZXB8PmRlZiBmKHNpbmdsZV9kaWdpdCk6Cjx8YXJnX3NlcHw+eyJzaW5nbGVfZGlnaXQiOiAiNSJ9)

<|begin\_of\_text|><|trace\_context\_start|>

def f(single\_digit):

result = []

for c in range(1, 11):

if c != single\_digit:

result.append(c)

return result

<|frame\_sep|><|inv\_return\_sep|>

<|src\_sep|> return result

<|arg\_sep|>"[1, 2, 3, 4, 6, 7, 8, 9, 10]"

<|action\_sep|><|inv\_step\_call|>

<|frame\_sep|>

----END OF PROMPT----

<|inv\_call\_sep|>

<|src\_sep|>def f(single\_digit):

<|arg\_sep|>{"single\_digit": "5"}

Figure B.4: CruxEval input prompt for the inv\_step\_call actions. Predictions are generated by the 1.8 B1.8\text{\,}\mathrm{B} parameter neural debugger trained on 150 B150\text{\,}\mathrm{B} debugger trace tokens. Some line breaks are inserted for illustration purposes.

## Appendix C Neural debugger trace example

In Figure [C.1](#A3.F1 "Figure C.1 ‣ Appendix C Neural debugger trace example ‣ Towards a Neural Debugger for Python"), we show the ground truth neural debugger trace with forward and inverse prediction.
The example is inspired by the interactive Python code debugging example with CWM (codgenteam2025\_cwm).
In this case, the inverse debugger trace in Figure [1(b)](#A3.F1.sf2 "Figure 1(b) ‣ Figure C.1 ‣ Appendix C Neural debugger trace example ‣ Towards a Neural Debugger for Python")is an example, where there are infinitely many possible s and t values for the return argument n=2.

[⬇](data:text/plain;base64,PHxiZWdpbl9vZl90ZXh0fD48fHRyYWNlX2NvbnRleHRfc3RhcnR8PgpkZWYgY291bnQocywgdCk6CiAgICBuID0gMAogICAgZm9yIGMgaW4gczoKICAgICAgICBuICs9IGludChjID09IHQpCiAgICByZXR1cm4gbgoKZGVmIG1haW4oKToKICAgIHJldHVybiBjb3VudCgiYmVycnkiLCAiciIpCjx8ZnJhbWVfc2VwfD48fGNhbGxfc2VwfD4KPHxzcmNfc2VwfD5kZWYgbWFpbigpOjx8YXJnX3NlcHw+e30KPHxhY3Rpb25fc2VwfD48fHN0ZXBfaW50b3w+Cgo8fGZyYW1lX3NlcHw+PHxsaW5lX3NlcHw+Cjx8YXJnX3NlcHw+e308fHNyY19zZXB8PiAgICByZXR1cm4gY291bnQoImJlcnJ5IiwgInIiKQo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGNhbGxfc2VwfD4KPHxzcmNfc2VwfD5kZWYgY291bnQocywgdCk6PHxhcmdfc2VwfD57InMiOiAiJ2JlcnJ5JyIsICJ0IjogIidyJyJ9Cjx8YWN0aW9uX3NlcHw+PHxzdGVwX2ludG98PgoKPHxmcmFtZV9zZXB8Pjx8bGluZV9zZXB8Pgo8fGFyZ19zZXB8PnsiLi4iOiAiLi4ifTx8c3JjX3NlcHw+ICAgIG4gPSAwCjx8YWN0aW9uX3NlcHw+PHxzdGVwX2ludG98PgoKPHxmcmFtZV9zZXB8Pjx8bGluZV9zZXB8Pgo8fGFyZ19zZXB8PnsiLi4iOiAiLi4iLCAibiI6ICIwIn08fHNyY19zZXB8PiAgICBmb3IgYyBpbiBzOgo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGxpbmVfc2VwfD4KPHxhcmdfc2VwfD57Ii4uIjogIi4uIiwgImMiOiAiJ2InIn08fHNyY19zZXB8PiAgICAgICAgbiArPSBpbnQoYyA9PSB0KQo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGxpbmVfc2VwfD4KPHxhcmdfc2VwfD57Ii4uIjogIi4uIn08fHNyY19zZXB8PiAgICBmb3IgYyBpbiBzOgo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGxpbmVfc2VwfD4KPHxhcmdfc2VwfD57Ii4uIjogIi4uIiwgImMiOiAiJ2UnIn08fHNyY19zZXB8PiAgICAgICAgbiArPSBpbnQoYyA9PSB0KQo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGxpbmVfc2VwfD4KPHxhcmdfc2VwfD57Ii4uIjogIi4uIn08fHNyY19zZXB8PiAgICBmb3IgYyBpbiBzOgo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGxpbmVfc2VwfD4KPHxhcmdfc2VwfD57Ii4uIjogIi4uIiwgImMiOiAiJ3InIn08fHNyY19zZXB8PiAgICAgICAgbiArPSBpbnQoYyA9PSB0KQo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGxpbmVfc2VwfD4KPHxhcmdfc2VwfD57Ii4uIjogIi4uIiwgIm4iOiAiMSJ9PHxzcmNfc2VwfD4gICAgZm9yIGMgaW4gczoKPHxhY3Rpb25fc2VwfD48fHN0ZXBfaW50b3w+Cgo8fGZyYW1lX3NlcHw+PHxsaW5lX3NlcHw+Cjx8YXJnX3NlcHw+eyIuLiI6ICIuLiJ9PHxzcmNfc2VwfD4gICAgICAgIG4gKz0gaW50KGMgPT0gdCkKPHxhY3Rpb25fc2VwfD48fHN0ZXBfaW50b3w+Cgo8fGZyYW1lX3NlcHw+PHxsaW5lX3NlcHw+Cjx8YXJnX3NlcHw+eyIuLiI6ICIuLiIsICJuIjogIjIifTx8c3JjX3NlcHw+ICAgIGZvciBjIGluIHM6Cjx8YWN0aW9uX3NlcHw+PHxzdGVwX2ludG98PgoKPHxmcmFtZV9zZXB8Pjx8bGluZV9zZXB8Pgo8fGFyZ19zZXB8PnsiLi4iOiAiLi4iLCAiYyI6ICIneScifTx8c3JjX3NlcHw+ICAgICAgICBuICs9IGludChjID09IHQpCjx8YWN0aW9uX3NlcHw+PHxzdGVwX2ludG98PgoKPHxmcmFtZV9zZXB8Pjx8bGluZV9zZXB8Pgo8fGFyZ19zZXB8PnsiLi4iOiAiLi4ifTx8c3JjX3NlcHw+ICAgIGZvciBjIGluIHM6Cjx8YWN0aW9uX3NlcHw+PHxzdGVwX2ludG98PgoKPHxmcmFtZV9zZXB8Pjx8bGluZV9zZXB8Pgo8fGFyZ19zZXB8PnsiLi4iOiAiLi4ifTx8c3JjX3NlcHw+ICAgIHJldHVybiBuCjx8YWN0aW9uX3NlcHw+PHxzdGVwX2ludG98PgoKPHxmcmFtZV9zZXB8Pjx8cmV0dXJuX3NlcHw+PHxzcmNfc2VwfD4gICAgcmV0dXJuIG4KPHxhcmdfc2VwfD4iMiIKPHxhY3Rpb25fc2VwfD48fHN0ZXBfaW50b3w+Cgo8fGZyYW1lX3NlcHw+PHxyZXR1cm5fc2VwfD4KPHxzcmNfc2VwfD4gICAgcmV0dXJuIGNvdW50KCJiZXJyeSIsICJyIik8fGFyZ19zZXB8PiIyIgo8fGFjdGlvbl9zZXB8Pjx8c3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGV4aXRfbm9ybWFsfD48fHRyYWNlX2VuZHw+PHxlbmRfb2ZfdGV4dHw+)

<|begin\_of\_text|><|trace\_context\_start|>

def count(s, t):

n = 0

for c in s:

n += int(c == t)

return n

def main():

return count("berry", "r")

<|frame\_sep|><|call\_sep|>

<|src\_sep|>def main():<|arg\_sep|>{}

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{}<|src\_sep|> return count("berry", "r")

<|action\_sep|><|step\_into|>

<|frame\_sep|><|call\_sep|>

<|src\_sep|>def count(s, t):<|arg\_sep|>{"s": "’berry’", "t": "’r’"}

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": ".."}<|src\_sep|> n = 0

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": "..", "n": "0"}<|src\_sep|> for c in s:

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": "..", "c": "’b’"}<|src\_sep|> n += int(c == t)

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": ".."}<|src\_sep|> for c in s:

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": "..", "c": "’e’"}<|src\_sep|> n += int(c == t)

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": ".."}<|src\_sep|> for c in s:

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": "..", "c": "’r’"}<|src\_sep|> n += int(c == t)

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": "..", "n": "1"}<|src\_sep|> for c in s:

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": ".."}<|src\_sep|> n += int(c == t)

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": "..", "n": "2"}<|src\_sep|> for c in s:

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": "..", "c": "’y’"}<|src\_sep|> n += int(c == t)

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": ".."}<|src\_sep|> for c in s:

<|action\_sep|><|step\_into|>

<|frame\_sep|><|line\_sep|>

<|arg\_sep|>{"..": ".."}<|src\_sep|> return n

<|action\_sep|><|step\_into|>

<|frame\_sep|><|return\_sep|><|src\_sep|> return n

<|arg\_sep|>"2"

<|action\_sep|><|step\_into|>

<|frame\_sep|><|return\_sep|>

<|src\_sep|> return count("berry", "r")<|arg\_sep|>"2"

<|action\_sep|><|step\_into|>

<|frame\_sep|><|exit\_normal|><|trace\_end|><|end\_of\_text|>

(a) Forward debugger trace.

[⬇](data:text/plain;base64,PHxiZWdpbl9vZl90ZXh0fD48fHRyYWNlX2NvbnRleHRfc3RhcnR8PgpkZWYgY291bnQocywgdCk6CiAgICBuID0gMAogICAgZm9yIGMgaW4gczoKICAgICAgICBuICs9IGludChjID09IHQpCiAgICByZXR1cm4gbgo8fGZyYW1lX3NlcHw+PHxpbnZfcmV0dXJuX3NlcHw+Cjx8c3JjX3NlcHw+ICAgIHJldHVybiBuPHxhcmdfc2VwfD4iMiIKPHxhY3Rpb25fc2VwfD48fGludl9zdGVwX2ludG98PgoKPHxmcmFtZV9zZXB8Pjx8aW52X2xpbmVfc2VwfD4KPHxzcmNfc2VwfD4gICAgcmV0dXJuIG48fGFyZ19zZXB8PnsiLi4iOiAiLi4ifQo8fGFjdGlvbl9zZXB8Pjx8aW52X3N0ZXBfaW50b3w+Cgo8fGZyYW1lX3NlcHw+PHxpbnZfbGluZV9zZXB8Pgo8fHNyY19zZXB8PiAgICBmb3IgYyBpbiBzOjx8YXJnX3NlcHw+eyIuLiI6ICIuLiJ9Cjx8YWN0aW9uX3NlcHw+PHxpbnZfc3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGludl9saW5lX3NlcHw+Cjx8c3JjX3NlcHw+ICAgICAgICBuICs9IGludChjID09IHQpPHxhcmdfc2VwfD57Ii4uIjogIi4uIn0KPHxhY3Rpb25fc2VwfD48fGludl9zdGVwX2ludG98PgoKPHxmcmFtZV9zZXB8Pjx8aW52X2xpbmVfc2VwfD4KPHxzcmNfc2VwfD4gICAgZm9yIGMgaW4gczo8fGFyZ19zZXB8PnsiLi4iOiAiLi4iLCAiYyI6ICIncicifQo8fGFjdGlvbl9zZXB8Pjx8aW52X3N0ZXBfaW50b3w+Cgo8fGZyYW1lX3NlcHw+PHxpbnZfbGluZV9zZXB8Pgo8fHNyY19zZXB8PiAgICAgICAgbiArPSBpbnQoYyA9PSB0KTx8YXJnX3NlcHw+eyIuLiI6ICIuLiIsICJuIjogIjEifQo8fGFjdGlvbl9zZXB8Pjx8aW52X3N0ZXBfaW50b3w+Cgo8fGZyYW1lX3NlcHw+PHxpbnZfbGluZV9zZXB8Pgo8fHNyY19zZXB8PiAgICBmb3IgYyBpbiBzOjx8YXJnX3NlcHw+eyIuLiI6ICIuLiJ9Cjx8YWN0aW9uX3NlcHw+PHxpbnZfc3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGludl9saW5lX3NlcHw+Cjx8c3JjX3NlcHw+ICAgICAgICBuICs9IGludChjID09IHQpPHxhcmdfc2VwfD57Ii4uIjogIi4uIiwgIm4iOiAiMCJ9Cjx8YWN0aW9uX3NlcHw+PHxpbnZfc3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGludl9saW5lX3NlcHw+Cjx8c3JjX3NlcHw+ICAgIGZvciBjIGluIHM6PHxhcmdfc2VwfD57Ii4uIjogIi4uIiwgImMiOiAiJ2UnIn0KPHxhY3Rpb25fc2VwfD48fGludl9zdGVwX2ludG98PgoKPHxmcmFtZV9zZXB8Pjx8aW52X2xpbmVfc2VwfD4KPHxzcmNfc2VwfD4gICAgICAgIG4gKz0gaW50KGMgPT0gdCk8fGFyZ19zZXB8PnsiLi4iOiAiLi4ifQo8fGFjdGlvbl9zZXB8Pjx8aW52X3N0ZXBfaW50b3w+Cgo8fGZyYW1lX3NlcHw+PHxpbnZfbGluZV9zZXB8Pgo8fHNyY19zZXB8PiAgICBmb3IgYyBpbiBzOjx8YXJnX3NlcHw+eyIuLiI6ICIuLiIsICJjIjogIidiJyJ9Cjx8YWN0aW9uX3NlcHw+PHxpbnZfc3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGludl9saW5lX3NlcHw+Cjx8c3JjX3NlcHw+ICAgICAgICBuICs9IGludChjID09IHQpPHxhcmdfc2VwfD57Ii4uIjogIi4uIn0KPHxhY3Rpb25fc2VwfD48fGludl9zdGVwX2ludG98PgoKPHxmcmFtZV9zZXB8Pjx8aW52X2xpbmVfc2VwfD4KPHxzcmNfc2VwfD4gICAgZm9yIGMgaW4gczo8fGFyZ19zZXB8PnsiLi4iOiAiLi4ifQo8fGFjdGlvbl9zZXB8Pjx8aW52X3N0ZXBfaW50b3w+Cgo8fGZyYW1lX3NlcHw+PHxpbnZfbGluZV9zZXB8Pgo8fHNyY19zZXB8PiAgICBuID0gMDx8YXJnX3NlcHw+eyIuLiI6ICIuLiJ9Cjx8YWN0aW9uX3NlcHw+PHxpbnZfc3RlcF9pbnRvfD4KCjx8ZnJhbWVfc2VwfD48fGludl9jYWxsX3NlcHw+Cjx8c3JjX3NlcHw+ZGVmIGNvdW50KHMsIHQpOjx8YXJnX3NlcHw+eyJzIjogIidiZXJyeSciLCAidCI6ICIncicifQo8fGFjdGlvbl9zZXB8Pjx8aW52X3N0ZXBfaW50b3w+Cgo8fGZyYW1lX3NlcHw+PHxpbnZfZXhpdF9lbnRyeXw+PHx0cmFjZV9lbmR8Pjx8ZW5kX29mX3RleHR8Pg==)

<|begin\_of\_text|><|trace\_context\_start|>

def count(s, t):

n = 0

for c in s:

n += int(c == t)

return n

<|frame\_sep|><|inv\_return\_sep|>

<|src\_sep|> return n<|arg\_sep|>"2"

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> return n<|arg\_sep|>{"..": ".."}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> for c in s:<|arg\_sep|>{"..": ".."}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> n += int(c == t)<|arg\_sep|>{"..": ".."}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> for c in s:<|arg\_sep|>{"..": "..", "c": "’r’"}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> n += int(c == t)<|arg\_sep|>{"..": "..", "n": "1"}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> for c in s:<|arg\_sep|>{"..": ".."}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> n += int(c == t)<|arg\_sep|>{"..": "..", "n": "0"}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> for c in s:<|arg\_sep|>{"..": "..", "c": "’e’"}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> n += int(c == t)<|arg\_sep|>{"..": ".."}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> for c in s:<|arg\_sep|>{"..": "..", "c": "’b’"}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> n += int(c == t)<|arg\_sep|>{"..": ".."}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> for c in s:<|arg\_sep|>{"..": ".."}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_line\_sep|>

<|src\_sep|> n = 0<|arg\_sep|>{"..": ".."}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_call\_sep|>

<|src\_sep|>def count(s, t):<|arg\_sep|>{"s": "’berry’", "t": "’r’"}

<|action\_sep|><|inv\_step\_into|>

<|frame\_sep|><|inv\_exit\_entry|><|trace\_end|><|end\_of\_text|>

(b) Inverse debugger trace.

Figure C.1: An example of a forward and inverse debugger trace for a function counting the occurrences of t in s. We take only step\_into or inv\_step\_into actions to visit every frame. While the forward debugger trace is deterministic given its inputs, the inverse debugger trace with return argument n=2 is an example with infinitely many input combinations for s and t.
