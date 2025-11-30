# NumericalO4LSP

Numerical Optimization for large scale problems





#### Modified Newton method (+ Back-tracking)

* dopo
* 


---
#### Truncated Newton (Newton–CG)  

Metodo per ottimizzazione numerica su larga scala basato su **Newton Inexacto** + **Gradiente Coniugato troncato (CG)** con **line search**.


#### Obiettivo
Trovare una direzione di Newton approssimata $p^k$ risolvendo  
$\nabla^2 f(x^k) \, p^k = -\nabla f(x^k)$
senza formare o fattorizzare l'Hessiana, utilizzando un algoritmo iterativo (CG).

Il metodo interrompe CG prima della convergenza completa (**truncated**) e gestisce anche il caso in cui l’Hessiana non sia definita positiva.


#### Struttura del metodo

##### 1. Sistema lineare da risolvere
$H_k p = -g_k \quad \text{con } H_k=\nabla^2 f(x^k),\ g_k=\nabla f(x^k)$

Si evita di calcolare $H_k$ esplicitamente: basta saper applicare  
$v\mapsto H_k v$


##### 2. Risoluzione tramite **CG troncato**
Durante CG si interrompe quando:
- $ \|H_k p_m + g_k\| \le \eta_k \|g_k\| $(accuratezza inexact Newton)  
- raggiunto un numero massimo di iterazioni  
- individuata **curvatura negativa**  
  $d_j^\top H_k d_j \le 0$
  → CG si ferma e si usa la direzione $d_j$ come direzione discendente.


##### 3. Direzione sempre di discesa
Assicurare:
$g_k^\top p^k < 0$
Se non avviene → usare $-g_k$ o la direzione negativa trovata da CG.


##### 4. Line Search (Backtracking)
Trovare $\alpha_k$ tale che:
$f(x^k + \alpha_k p^k) \le f(x^k) + c \alpha_k g_k^\top p^k$

Aggiornamento:
$x^{k+1} = x^k + \alpha_k p^k$


