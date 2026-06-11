# -*- coding: utf-8 -*-
"""Roda todos os exercícios do Cap. 12 em sequência (figuras salvas em figs/)."""
import ex1_impulso
import ex2_degrau
import ex3a_planta_zoh
import ex3b_pi_euler
import ex3c_regulacao_robusta
import ex3d_estab_discreta
import ex3e_euler_backward_tustin
import ex3f_rastreamento_euler
import ex3g_rastreamento_discreto

for mod in [ex1_impulso, ex2_degrau, ex3a_planta_zoh, ex3b_pi_euler,
            ex3c_regulacao_robusta, ex3d_estab_discreta,
            ex3e_euler_backward_tustin, ex3f_rastreamento_euler,
            ex3g_rastreamento_discreto]:
    print("\n" + "=" * 70)
    print(f"== {mod.__name__}")
    print("=" * 70)
    mod.main()
