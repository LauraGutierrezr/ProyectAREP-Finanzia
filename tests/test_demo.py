"""Pruebas automáticas (pytest). Ejecutar después del pipeline:  PYTHONPATH=src pytest -q"""
import pytest

from fi import config, recommender, store
from fi.assistant import assistant, llm, router, verifier


def test_router_intenciones():
    assert router.clasificar("¿En qué estoy gastando más?") == "mayor_gasto"
    assert router.clasificar("¿Qué gastos están aumentando?") == "aumentos"
    assert router.clasificar("¿Cuánto voy a gastar el próximo mes?") == "pronostico"
    assert router.clasificar("¿Hay algún cobro raro?") == "inusual"
    assert router.clasificar("¿Por qué no llego a mi meta?") == "meta"
    assert router.clasificar("¿Qué puedo hacer para mejorar?") == "accion"


@pytest.mark.parametrize("q", ["¿En qué fondo invierto?", "¿Compro bitcoin?", "¿Pido un préstamo?"])
def test_router_fuera_de_alcance(q):
    assert router.clasificar(q) == "fuera_alcance"


def test_verificador_detecta_cifra_inventada():
    ctx = [{"id": "f1", "monto": "$200.000"}, {"id": "f2", "monto": "$100.000"}]
    assert verifier.verificar("Necesitas $200.000 [f1] y ahorras $100.000 [f2].", ctx)["ok"]
    malo = verifier.verificar("Necesitas $250.000 [f1].", ctx)
    assert not malo["ok"] and malo["no_respaldadas"] == ["$250.000"]
    assert not verifier.verificar("Ahorras $100.000 [f9].", ctx)["ok"]          # cita inexistente
    assert verifier.verificar("Son 0,1 millones [f2].", ctx)["ok"]              # misma cifra en otra unidad


def _usuario():
    return store.datos()["usuarios"].index[0]


def test_recomendaciones_respetan_limites():
    for uid in store.datos()["usuarios"].index[:30]:
        r = recommender.recomendar(uid)
        assert len(r["recomendaciones"]) <= 3
        est = store.estadisticas_categoria(uid, config.MES_DEMO)
        for rec in r["recomendaciones"]:
            if rec["regla"] in ("R1", "R3"):
                assert rec["categoria"] in config.DISCRECIONALES
                assert rec["objetivo"] >= est.loc[rec["categoria"], "p25_12m"] - 1


def test_asistente_plantilla_siempre_fundamentado():
    for q in ["¿En qué estoy gastando más?", "¿Por qué no llego a mi meta?", "¿Qué puedo hacer para mejorar?"]:
        r = assistant.responder(_usuario(), q, "plantilla")
        assert r["verificacion"]["ok"], r["respuesta"]


def test_asistente_respalda_con_plantilla_si_el_llm_inventa(monkeypatch):
    monkeypatch.setattr(llm, "generar", lambda *a, **k: "Este mes ahorrarás $999.999 [f2].")
    r = assistant.responder(_usuario(), "¿Cómo voy con mi meta?", "anthropic")
    assert r["intentos"] == 2 and r["respaldo_plantilla"] and r["verificacion"]["ok"]
