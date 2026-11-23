"""Respuestas con plantilla determinista (condición C0 y respaldo cuando el LLM falla la verificación)."""
from __future__ import annotations


def _h(ctx, desc_inicio):
    return [h for h in ctx if h["descripcion"].startswith(desc_inicio)]


def redactar(intencion: str, ctx: list[dict]) -> str:
    mes = ctx[0]
    if intencion == "mayor_gasto":
        tot = _h(ctx, "Gasto total")[0]
        cats = _h(ctx, "Categoría con mayor gasto")
        partes = ", ".join(f"{c['categoria'].lower()} con {c['monto']} ({c['porcentaje']}) [{c['id']}]" for c in cats)
        return (f"En {mes['ultimo_mes_cerrado'].split(' de ')[0]} gastaste {tot['monto']} en total [{tot['id']}]. "
                f"Donde más se fue tu dinero fue en {partes}.")
    if intencion == "aumentos":
        cats = _h(ctx, "Categoría que aumentó")
        if not cats:
            return "Buenas noticias: este último mes ninguna categoría está por encima de lo que acostumbras gastar [f2]."
        partes = "; ".join(f"{c['categoria'].lower()}: {c['gasto_ultimo_mes']} frente a tu mediana de {c['mediana_6_meses']}, "
                           f"un {c['aumento']} más [{c['id']}]" for c in cats)
        return f"Comparado con tus últimos seis meses, subieron estos gastos: {partes}."
    if intencion == "pronostico":
        tot = _h(ctx, "Gasto total previsto")[0]
        prom = _h(ctx, "Promedio de gasto")[0]
        cats = _h(ctx, "Categoría con mayor gasto previsto")
        partes = ", ".join(f"{c['categoria'].lower()} ({c['monto']}) [{c['id']}]" for c in cats)
        return (f"Para {tot['mes']} el modelo estima un gasto de {tot['monto']} [{tot['id']}], frente a un promedio de "
                f"{prom['monto']} en los últimos tres meses [{prom['id']}]. Lo que más pesará: {partes}.")
    if intencion == "inusual":
        movs = _h(ctx, "Movimiento marcado")
        if not movs:
            return "No encontré movimientos fuera de lo habitual este mes [f2]."
        partes = " ".join(f"El {m['fecha']} hubo un cargo de {m['monto']} en {m['comercio']} ({m['categoria'].lower()}); "
                          f"tu gasto típico ahí es {m['gasto_tipico']} [{m['id']}]." for m in movs)
        cant = _h(ctx, "Cantidad de movimientos")[0]
        return f"Encontré {cant['cantidad']} movimiento(s) que vale la pena revisar [{cant['id']}]. {partes}"
    if intencion == "meta":
        meta = _h(ctx, "Meta de ahorro")[0]
        ah = _h(ctx, "Ahorro proyectado")[0]
        br = _h(ctx, "Brecha")
        cat = _h(ctx, "Categoría discrecional")[0]
        if br and ah["monto"].startswith("-"):
            txt = (f"Tu meta pide ahorrar {meta['aporte_mensual_necesario']} al mes [{meta['id']}], pero este mes el gasto previsto "
                   f"supera tu ingreso (ahorro proyectado de {ah['monto']}) [{ah['id']}]: te faltan {br[0]['monto']} [{br[0]['id']}]. ")
        elif br:
            txt = (f"Tu meta pide ahorrar {meta['aporte_mensual_necesario']} al mes [{meta['id']}] y este mes vas a ahorrar "
                   f"{ah['monto']} [{ah['id']}]: te faltan {br[0]['monto']} [{br[0]['id']}]. ")
        else:
            s = _h(ctx, "Sin brecha")[0]
            txt = (f"Vas bien: tu meta pide {meta['aporte_mensual_necesario']} al mes [{meta['id']}] y vas a ahorrar "
                   f"{ah['monto']} [{ah['id']}], {s['excedente']} de más [{s['id']}]. ")
        return txt + (f"La mayor diferencia está en {cat['categoria'].lower()}: se prevén {cat['previsto']} frente a una "
                      f"mediana de {cat['mediana_6_meses']} [{cat['id']}].")
    if intencion == "accion":
        recs = _h(ctx, "Recomendación")
        if not recs:
            return "No tengo recomendaciones para este mes: tu gasto está dentro de lo habitual."
        txt = " ".join(f"• {r['texto']} [{r['id']}]" for r in recs)
        con = _h(ctx, "Meses para cumplir la meta aplicando")
        if con:
            txt += f" Con estas acciones cumplirías tu meta en unos {con[0]['meses']} meses [{con[0]['id']}]."
        return "Esto es lo que te propongo: " + txt
    return ""
