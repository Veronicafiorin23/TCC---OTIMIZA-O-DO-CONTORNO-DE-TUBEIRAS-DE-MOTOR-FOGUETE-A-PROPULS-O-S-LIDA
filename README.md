# TCC---OTIMIZA-O-DO-CONTORNO-DE-TUBEIRAS-DE-MOTOR-FOGUETE-A-PROPULS-O-S-LIDA
#OTIMIZAÇÃO DO CONTORNO DE TUBEIRAS DE MOTOR-FOGUETE A PROPULSÃO SÓLIDA: MODELAGEM,  SIMULAÇÃO E ANÁLISE AVANÇADA.

#---------------------------------------------Codigo---------------------------------------------
# nozzle_opt_with_TOP_v6.py
"""
v6: mostra as FUNCOES que geram cada perfil (original, TOP, spline otimizada, MOC).
Base robusta (fallback isentropico) para evitar travamentos.
Imprime as expressões polinomiais no terminal e salva arquivos com coeficientes.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq, newton
from math import sqrt
import csv, os, time, traceback
import matplotlib.pyplot as plt
import textwrap

# ----------------- USER INPUTS -----------------
T0_chamber = 1520.0
p0_chamber = 2.7e6
R_gas = 208.31
gamma = 1.1361

mass_propellant = 8.4
burn_time = 5.7
mdot = mass_propellant / burn_time

d_throat = 26.0e-3
d_exit = 54.20e-3
rg = d_throat/2.0
r_exit = d_exit/2.0

ang_div_deg = 12.0
ang_div_rad = np.deg2rad(ang_div_deg)

L_div_initial = (r_exit - rg)/np.tan(ang_div_rad)

p_ambient = 101325.0
k_loss = 0.03

# resolution (keep small for debug; aumente se quiser)
n_ctrl = 8 # número de pontos de controle para otimização
N_grid = 151 # número de pontos na malha completa para simulação
N_char_moc = 12 # número de pontos na malha para MOC

ts = time.strftime("%Y%m%d_%H%M%S")
out_profile_csv = f'nozzle_optimized_profile_{ts}.csv'
moc_csv = f'nozzle_moc_profile_{ts}.csv'
spline_coeffs_file = f'spline_piecewise_coeffs_{ts}.txt'
moc_coeffs_file = f'moc_piecewise_coeffs_{ts}.txt'
summary_file = f'top_and_spline_summary_{ts}.txt'
logfile = f'nozzle_opt_log_{ts}.txt'

# ----------------- logging -----------------
def log(msg):
    print(msg)
    try:
        with open(logfile,'a') as f:
            f.write(msg + "\n")
    except Exception:
        pass

# ----------------- thermo helpers -----------------
def isentropic_p_from_p0(p0, M, gamma):
    return p0*(1+0.5*(gamma-1)*M**2)**(-gamma/(gamma-1))

def prandtl_meyer(M, gamma):
    if M <= 1.0: return 0.0
    return np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt((gamma-1)/(gamma+1)*(M**2-1))) - np.arctan(np.sqrt(M**2-1))

def area_mach_ratio(M, gamma):
    term = (1 + 0.5*(gamma-1)*M**2)
    exp = (gamma+1)/(2*(gamma-1))
    return (1.0/M) * term**exp

def solve_M_from_area_ratio(AoAstar, gamma, supersonic=True):
    if AoAstar <= 0:
        return np.nan
    lo, hi = (1.0001, 300.0) if supersonic else (1e-6, 0.9999)
    f = lambda M: area_mach_ratio(M, gamma) - AoAstar
    try:
        return brentq(f, lo, hi, maxiter=200)
    except Exception:
        return 2.0 if supersonic else 0.5

# ----------------- integrate_mach (robust) -----------------
def integrate_mach_from_throat(x_nodes, A_nodes, gamma):
    if np.any(A_nodes <= 0) or len(x_nodes) < 2:
        return x_nodes, np.full_like(x_nodes, np.nan, dtype=float)
    try:
        A_spline = CubicSpline(x_nodes, A_nodes, bc_type='natural')
    except Exception as e:
        log(f"[integrate_mach] spline fail: {e}")
        return x_nodes, np.full_like(x_nodes, np.nan, dtype=float)

    def dMdx(x,y):
        M = float(y[0])
        A = float(A_spline(x)); dAdx = float(A_spline.derivative()(x))
        denom = 1 - M*M
        if abs(denom) < 1e-8: denom = np.sign(denom)*1e-8
        return [(M/denom)*(dAdx/A)]

    x0 = float(x_nodes[0]); x1 = float(x_nodes[1]) if len(x_nodes)>1 else x0+1e-6
    eps = max(1e-9, 1e-6*max(1.0, abs(x_nodes[-1]-x_nodes[0])))
    t0 = x0 + min(eps, 0.5*(x1-x0) if x1>x0 else eps)
    t_final = float(x_nodes[-1])
    tol = 1e-12
    t_eval = np.array([xx for xx in x_nodes if (xx >= t0 - tol) and (xx <= t_final + tol)], dtype=float)
    if t_eval.size == 0:
        return x_nodes, np.full_like(x_nodes, np.nan, dtype=float)

    M_init = 1.0 + 1e-3
    sol = None
    for method in ('RK45','LSODA','RK23'):
        try:
            sol = solve_ivp(dMdx, (t0, t_final), [M_init], t_eval=t_eval, method=method,
                            max_step=max((t_final - t0)/200.0, 1e-6))
            if sol.status >= 0 and not np.any(np.isnan(sol.y[0])):
                break
        except Exception:
            sol = None
    if sol is None or sol.status < 0 or np.any(np.isnan(sol.y[0])):
        log("[integrate_mach] solver failed -> returning NaNs")
        return x_nodes, np.full_like(x_nodes, np.nan, dtype=float)

    M_full = np.full(len(x_nodes), np.nan, dtype=float)
    idxs = [i for i, xx in enumerate(x_nodes) if (xx >= t0 - tol) and (xx <= t_final + tol)]
    if len(idxs) == len(sol.t):
        M_full[idxs] = sol.y[0]
    else:
        M_full[idxs] = np.interp([x_nodes[i] for i in idxs], sol.t, sol.y[0])
    return x_nodes, M_full

# ----------------- thrust sim (kept for fallback if needed) -----------------
def compute_thrust_with_possible_shock(x_full, r_full, p0, T0, mdot, gamma, R, p_amb, k_loss, find_shock=True):
    if np.any(r_full <= 0):
        return {'success':False}
    A_full = np.pi * r_full**2
    L = x_full[-1]
    p0_prof = p0 * np.exp(-k_loss*(x_full - x_full[0])/(L+1e-12))

    xs_iso, M_iso = integrate_mach_from_throat(x_full, A_full, gamma)
    if not np.any(np.isnan(M_iso)):
        p_static = [isentropic_p_from_p0(p0_prof[i], M_iso[i], gamma) for i in range(len(x_full))]
        Te = T0/(1+0.5*(gamma-1)*M_iso[-1]**2)
        ve = M_iso[-1]*np.sqrt(gamma*R*Te)
        F_iso = mdot*ve + (p_static[-1]-p_amb)*A_full[-1]
        base={'mode':'no_shock','F':F_iso,'details':{'M':M_iso,'p_static':p_static,'ve':ve,'pe':p_static[-1]},'shock_idx':None}
    else:
        base={'mode':'fail','F':None}

    if not find_shock:
        return {'success': base['F'] is not None, **base}

    best=None; best_diff=1e12
    for idx in range(2, len(x_full)-4):
        xs_up, M_up = integrate_mach_from_throat(x_full[:idx+1], A_full[:idx+1], gamma)
        if np.any(np.isnan(M_up)): continue
        M1 = M_up[-1]
        if M1 <= 1.0: continue
        # normal shock
        p2_p1 = 1+(2*gamma/(gamma+1))*(M1**2 - 1)
        M2_sq = (1+0.5*(gamma-1)*M1**2)/(gamma*M1**2 - 0.5*(gamma-1))
        M2 = sqrt(M2_sq) if M2_sq>0 else 1e-6
        term1 = (1+0.5*(gamma-1)*M1**2)**(gamma/(gamma-1))
        term2 = (1+0.5*(gamma-1)*M2**2)**(gamma/(gamma-1))
        p02p01 = (1/p2_p1)*(term1/term2)

        A_spline = CubicSpline(x_full, A_full, bc_type='natural')
        def dMdx2(x,y):
            M = float(y[0]); A = float(A_spline(x)); dAdx = float(A_spline.derivative()(x))
            denom = 1 - M*M
            if abs(denom) < 1e-8: denom = np.sign(denom)*1e-8
            return [(M/denom)*(dAdx/A)]
        x_down = x_full[idx:]
        try:
            sol = solve_ivp(dMdx2, (x_down[0]+1e-9, x_down[-1]), [M2], t_eval=x_down, method='RK45', max_step=max((x_down[-1]-x_down[0])/200.0, 1e-5))
            M_down = sol.y[0]
            if np.any(np.isnan(M_down)): continue
        except Exception:
            continue

        L = x_full[-1]
        p0_after = p0_prof[idx] * p02p01
        p_static = np.zeros_like(x_full)
        for i in range(len(x_full)):
            if i <= idx:
                p_static[i] = isentropic_p_from_p0(p0_prof[i], M_up[i], gamma)
            else:
                x_rel = (x_full[i]-x_full[idx])/(L+1e-12)
                p0_loc = p0_after*np.exp(-k_loss*x_rel)
                p_static[i] = isentropic_p_from_p0(p0_loc, M_down[i-idx], gamma)
        Te = T0/(1+0.5*(gamma-1)*M_down[-1]**2)
        ve = M_down[-1]*np.sqrt(gamma*R*Te)
        pe = p_static[-1]
        F = mdot*ve + (pe-p_amb)*A_full[-1]
        diff = abs(pe - p_amb)
        if diff < best_diff:
            best_diff = diff
            best={'mode':'shock','F':F,'details':{'M':np.concatenate([M_up[:-1],M_down]),'p_static':p_static,'ve':ve,'pe':pe},'shock_idx':idx}

    if best and base['F'] is not None:
        if best['F'] >= base['F']*0.999:
            return {'success':True, **best}
    if base['F'] is not None:
        return {'success':True, **base}
    return {'success':False}

# ----------------- fallback (isentropic inversion A->M) -----------------
def compute_thrust_isentropic_fallback(x_full, r_full, p0, T0, mdot, gamma, R, p_amb, k_loss):
    A_throat = np.pi * (rg**2)
    A_full = np.pi * r_full**2
    AoAstar = A_full / (A_throat + 1e-20)
    M_arr = np.zeros_like(AoAstar)
    for i, val in enumerate(AoAstar):
        if val <= 0:
            M_arr[i] = np.nan
            continue
        try:
            M_guess = solve_M_from_area_ratio(val, gamma, supersonic=True)
            if M_guess <= 1.001:
                M_guess = solve_M_from_area_ratio(val, gamma, supersonic=False)
        except Exception:
            M_guess = 1.0
        M_arr[i] = M_guess
    if np.any(np.isnan(M_arr)):
        return {'success':False}
    L = x_full[-1]
    p0_prof = p0 * np.exp(-k_loss*(x_full - x_full[0])/(L+1e-12))
    p_static = [isentropic_p_from_p0(p0_prof[i], M_arr[i], gamma) for i in range(len(x_full))]
    Te = T0/(1+0.5*(gamma-1)*M_arr[-1]**2)
    ve = M_arr[-1]*np.sqrt(gamma*R*Te)
    F_iso = mdot*ve + (p_static[-1]-p_amb)*(A_full[-1])
    return {'success':True,'mode':'no_shock_fallback','F':F_iso,'details':{'M':M_arr,'p_static':p_static,'ve':ve,'pe':p_static[-1]},'shock_idx':None}

# ----------------- objective for optimization -----------------
def objective_ctrl_points(y_ctrl, x_ctrl, x_full, rg, r_exit, p0, T0, mdot, gamma, R, p_amb, k_loss):
    xs = np.concatenate(([0.0], x_ctrl, [x_full[-1]]))
    ys = np.concatenate(([rg], y_ctrl, [r_exit]))
    try:
        spl = CubicSpline(xs, ys, bc_type='natural')
        r_full = spl(x_full)
    except Exception:
        return 1e6
    if np.any(r_full <= 0): return 1e6
    sim = compute_thrust_with_possible_shock(x_full, r_full, p0, T0, mdot, gamma, R, p_amb, k_loss)
    if not sim.get('success',False) or sim.get('F') is None:
        sim_fb = compute_thrust_isentropic_fallback(x_full, r_full, p0, T0, mdot, gamma, R, p_amb, k_loss)
        if sim_fb.get('success',False) and sim_fb.get('F') is not None:
            return -sim_fb['F']
        return 1e6
    return -sim['F']

# ----------------- MOC approx (robusto e simples) -----------------
def design_moc_nozzle(r_throat, r_exit, L_estimate, gamma, N_char=12):
    AeAstar = (r_exit / r_throat)**2
    try:
        Me = solve_M_from_area_ratio(AeAstar, gamma, supersonic=True)
    except Exception:
        Me = 2.0
    # construir perfil simples por interpolacao area
    x_wall = np.linspace(0.0, L_estimate, max(10, N_char))
    r_wall = np.zeros_like(x_wall)
    A_star = np.pi * r_throat**2
    for i,x in enumerate(x_wall):
        A_x = A_star + (np.pi*r_exit**2 - A_star) * (x / (L_estimate+1e-12))
        r_wall[i] = np.sqrt(max(A_x/np.pi, 0.0))
    r_wall[-1] = r_exit
    return x_wall, r_wall, Me

# ----------------- helper: expand spline piece to global polynomial -----------------
def expand_piece_from_cs(cs, x_knots, i):
    # cs shape: (4, n_intervals) as returned by CubicSpline.c
    c0,c1,c2,c3 = cs[:,i]
    xi = x_knots[i]
    # piece is c0 + c1*(x-xi) + c2*(x-xi)^2 + c3*(x-xi)^3
    # expand to a*x^3 + b*x^2 + c*x + d
    a = c3
    b = -3*c3*xi + c2
    ccoef = 3*c3*xi*xi - 2*c2*xi + c1
    dcoef = c0 - c1*xi + c2*xi*xi - c3*xi**3
    return a,b,ccoef,dcoef

# ----------------- main -----------------
def main():
    start_time = time.time()  # Captura o tempo inicial

    if os.path.exists(logfile):
        try: os.remove(logfile)
        except: pass
    log("### nozzle_opt_with_TOP_v6 run ###")

    x_full = np.linspace(0.0, L_div_initial, N_grid)
    x_ctrl = np.linspace(0.25*L_div_initial, 0.9*L_div_initial, n_ctrl)
    y_ctrl0 = rg + (r_exit - rg)*((x_ctrl/L_div_initial)**0.8)
    xs_opt_guess = np.concatenate(([0.0], x_ctrl, [L_div_initial]))
    ys_opt_guess = np.concatenate(([rg], y_ctrl0, [r_exit]))
    spline_init = CubicSpline(xs_opt_guess, ys_opt_guess, bc_type='natural')
    r_init_guess = spline_init(x_full)

    base = compute_thrust_with_possible_shock(x_full, r_init_guess, p0_chamber, T0_chamber, mdot, gamma, R_gas, p_ambient, k_loss)
    log(f"Baseline sim success? {base.get('success',False)}")
    if not base.get('success',False):
        fb = compute_thrust_isentropic_fallback(x_full, r_init_guess, p0_chamber, T0_chamber, mdot, gamma, R_gas, p_ambient, k_loss)
        if fb.get('success',False):
            log(f"Fallback baseline F ≈ {fb['F']:.6f} N (apenas para debug).")
        else:
            log("Baseline e fallback falharam — geometria salva em debug_geometry_baseline.csv")
            with open('debug_geometry_baseline.csv','w',newline='') as f:
                w = csv.writer(f); w.writerow(['x','r'])
                for xi,ri in zip(x_full, r_init_guess):
                    w.writerow([float(xi), float(ri)])

    # optimization
    bounds = [(rg, max(r_exit*1.15, rg*1.01))]*n_ctrl
    try:
        res = minimize(objective_ctrl_points, y_ctrl0,
                       args=(x_ctrl, x_full, rg, r_exit, p0_chamber, T0_chamber, mdot, gamma, R_gas, p_ambient, k_loss),
                       method='SLSQP', bounds=bounds,
                       options={'maxiter':80,'ftol':1e-6, 'disp': False})
        log(f"Otimização: success={res.success}, message='{res.message}'")
    except Exception as e:
        log(f"Otimização gerou excecao: {e}")
        res = None

    if res is not None and res.success:
        y_opt = res.x
    else:
        log("Usando palpite inicial (otimização nao convergiu).")
        y_opt = y_ctrl0

    xs_opt = np.concatenate(([0.0], x_ctrl, [L_div_initial]))
    ys_opt = np.concatenate(([rg], y_opt, [r_exit]))
    spline_opt = CubicSpline(xs_opt, ys_opt, bc_type='natural')
    r_opt = spline_opt(x_full)

    opt_sim = compute_thrust_with_possible_shock(x_full, r_opt, p0_chamber, T0_chamber, mdot, gamma, R_gas, p_ambient, k_loss)
    if not opt_sim.get('success',False):
        log("Sim otimizado numeric falhou -> fallback isentropico.")
        opt_sim = compute_thrust_isentropic_fallback(x_full, r_opt, p0_chamber, T0_chamber, mdot, gamma, R_gas, p_ambient, k_loss)

    # TOP cubic
    try:
        theta_throat = 0.0
        theta_exit = np.tan(ang_div_rad)
        L = L_div_initial; d = rg; c = theta_throat
        A_mat = np.array([[L**3, L**2],[3*L**2, 2*L]])
        rhs = np.array([r_exit - c*L - d, theta_exit - c])
        a_top, b_top = np.linalg.solve(A_mat, rhs)
        def y_top(x): return a_top*x**3 + b_top*x**2 + c*x + d
        y_top_full = y_top(x_full)
    except Exception:
        y_top_full = rg + (r_exit - rg)*(x_full/(L_div_initial+1e-12))

    # MOC approx
    x_moc, r_moc, Me = design_moc_nozzle(rg, r_exit, L_div_initial, gamma, N_char=N_char_moc)
    try:
        if len(x_moc) >= 2:
            x_moc_new = np.linspace(0.0, L_div_initial, max(40, len(x_moc)))
            r_moc = np.interp(x_moc_new, x_moc, r_moc)
            x_moc = x_moc_new
    except Exception:
        x_moc = np.array([0.0, L_div_initial]); r_moc = np.array([rg, r_exit])

    # save profiles
    with open(out_profile_csv,'w',newline='') as f:
        w = csv.writer(f); w.writerow(['x_m','r_m'])
        for xi,ri in zip(x_full, r_opt):
            w.writerow([float(xi), float(ri)])
    log(f"Perfil otimizado salvo em {out_profile_csv}")

    with open(moc_csv,'w',newline='') as f:
        w = csv.writer(f); w.writerow(['x_m','r_m'])
        for xi,ri in zip(x_moc, r_moc):
            w.writerow([float(xi), float(ri)])
    log(f"Perfil MOC salvo em {moc_csv}")

    # ---- export spline optimized piecewise coefficients ----
    cs = spline_opt.c
    x_knots = spline_opt.x
    try:
        n_intervals = cs.shape[1]
    except Exception:
        n_intervals = 0

    with open(spline_coeffs_file, 'w') as f:
        f.write("Spline otimizada - Piecewise cubic coefficients (per interval), expanded to ax^3+bx^2+cx+d\n")
        for i in range(n_intervals):
            a,b,cc,d = expand_piece_from_cs(cs, x_knots, i)
            xi = x_knots[i]; xi1 = x_knots[i+1]
            f.write(f"Interval {i}: [{xi:.6e}, {xi1:.6e}] -> a={a:.12e}, b={b:.12e}, c={cc:.12e}, d={d:.12e}\n")
    log(f"Spline piecewise coefficients salvos em {spline_coeffs_file}")

    # ---- build spline for MOC and export its piecewise coeffs ----
    try:
        spline_moc = CubicSpline(x_moc, r_moc, bc_type='natural')
        cs_m = spline_moc.c; xk_m = spline_moc.x
        with open(moc_coeffs_file, 'w') as f:
            f.write("MOC spline - Piecewise cubic coefficients (expanded)\n")
            for i in range(cs_m.shape[1]):
                a,b,cc,d = expand_piece_from_cs(cs_m, xk_m, i)
                f.write(f"Interval {i}: [{xk_m[i]:.6e}, {xk_m[i+1]:.6e}] -> a={a:.12e}, b={b:.12e}, c={cc:.12e}, d={d:.12e}\n")
        log(f"MOC piecewise coefficients salvos em {moc_coeffs_file}")
    except Exception as e:
        log(f"Falha ao exportar MOC spline coefficients: {e}")

    # ---- prepare human-readable function strings ----
    # original (linear)
    if L_div_initial != 0:
        a_lin = (r_exit - rg)/L_div_initial
        b_lin = rg
        orig_str = f"r_orig(x) = ({a_lin:.6e})*x + ({b_lin:.6e})"
    else:
        orig_str = f"r_orig(x) = constant {rg:.6e}"

    # TOP cubic
    top_str = f"r_TOP(x) = ({a_top:.6e})*x^3 + ({b_top:.6e})*x^2 + ({c:.6e})*x + ({d:.6e})"

    # Spline optimized - show first piece + list intervals count; full saved in file
    if n_intervals > 0:
        a1,b1,c1_coef,d1 = expand_piece_from_cs(cs, x_knots, 0)
        spline_first_str = f"Spline_opt first interval S0(x) = ({a1:.6e})*x^3 + ({b1:.6e})*x^2 + ({c1_coef:.6e})*x + ({d1:.6e}) on [{x_knots[0]:.6e},{x_knots[1]:.6e}]"
        spline_summary = f"Spline_opt has {n_intervals} intervals. Full coefficients in {spline_coeffs_file}."
    else:
        spline_first_str = "Spline_opt: no intervals"
        spline_summary = "Spline_opt: no coefficients"

    # MOC spline summary (first piece)
    try:
        cs_m = spline_moc.c; xk_m = spline_moc.x
        a_m0,b_m0,c_m0,d_m0 = expand_piece_from_cs(cs_m, xk_m, 0)
        moc_first_str = f"MOC spline first interval: ({a_m0:.6e})*x^3 + ({b_m0:.6e})*x^2 + ({c_m0:.6e})*x + ({d_m0:.6e}) on [{xk_m[0]:.6e},{xk_m[1]:.6e}]"
        moc_summary = f"MOC spline has {cs_m.shape[1]} intervals. Full coefficients in {moc_coeffs_file}."
    except Exception:
        moc_first_str = "MOC spline: unavailable"
        moc_summary = "MOC spline: unavailable"

    # ---- print functions to terminal ----
    print("\n=== FUNÇÕES QUE GERAM CADA PAREDE ===")
    print("1) Perfil ORIGINAL (cônico - linha):")
    print("   ", orig_str)
    print("\n2) TOP (cúbica Rao-approx):")
    print("   ", top_str)
    print("\n3) Spline OTIMIZADA (mostro 1º trecho; todas salvas em file):")
    print("   ", spline_first_str)
    print("   ", spline_summary)
    print("\n4) MoC (Método das Características) - aproximação:")
    print("   ", moc_first_str)
    print("   ", moc_summary)
    print("=====================================\n")

    # ---- plot with abbreviated function annotations ----
    plt.figure(figsize=(11,5.5))
    plt.plot([0.0, L_div_initial], [rg, r_exit], label='Perfil original (cônico)', color='tab:blue', linestyle='--', linewidth=2.2)
    plt.plot(x_full, r_opt, label='Perfil otimizado (spline)', color='tab:orange', linewidth=2.4)
    plt.plot(x_full, y_top_full, label='TOP (Rao-approx)', color='tab:green', linestyle=':', linewidth=2.0)
    plt.plot(x_moc, r_moc, label='MoC (Método das Características) - approx', color='tab:purple', linewidth=2.0)
    plt.scatter(np.concatenate(([0.0], x_ctrl, [L_div_initial])), np.concatenate(([rg], y_opt, [r_exit])), s=40, color='orange', edgecolor='k', label='ctrl pts otimizados')
    plt.xlabel('x (m) desde garganta'); plt.ylabel('raio r(x) (m)')
    plt.grid(True); plt.legend(loc='lower right')

    # abbreviated annotation (full coeff files saved)
    ann = "Original: r(x)=a x + b\nTOP: cúbico (completo em resumo)\nSpline_opt: primeiro trecho exibido no terminal e o trecho completo no arquivo.\nMOC spline: primeiro trecho exibido no terminal e o trecho completo no arquivo."
    plt.gca().text(0.02,0.95, ann, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout(); plt.show()

    # final report print
    print("\nArquivos gerados (contendo as funções completas):")
    print(" - Spline optimized piecewise:", spline_coeffs_file)
    print(" - MOC piecewise:", moc_coeffs_file)
    print(" - Optimized profile CSV:", out_profile_csv)
    print(" - MOC profile CSV:", moc_csv)
    print(" - Log:", logfile)
    print("\nSe quiser que eu mostre *todas* as peças aqui no terminal, diga que eu imprimo (pode ficar muito longo).")

    # Finalizar e calcular o tempo total
    end_time = time.time()  # Captura o tempo final
    elapsed_time = end_time - start_time
    print(f"\nTempo total de execução: {elapsed_time:.2f} segundos")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Erro durante execucao:")
        traceback.print_exc()
