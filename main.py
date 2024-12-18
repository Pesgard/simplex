from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import numpy as np

app = FastAPI(title="Simplex Method API", version="1.0")

class SimplexRequest(BaseModel):
    c: List[float] = Field(..., description="Coeficientes de la función objetivo.")
    A: List[List[float]] = Field(..., description="Matriz de restricciones.")
    b: List[float] = Field(..., description="Vector del lado derecho de las restricciones.")
    signos: List[str] = Field(..., description="Lista de signos de las restricciones ('<=', '>=', '=').")
    operacion: str = Field(..., description="Tipo de operación ('max' o 'min').")

    @validator('signos', each_item=True)
    def validar_signos(cls, v):
        if v not in ("<=", ">=", "="):
            raise ValueError("Los signos deben ser '<=', '>=', o '='.")
        return v

    @validator('operacion')
    def validar_operacion(cls, v):
        if v not in ("max", "min"):
            raise ValueError("La operación debe ser 'max' o 'min'.")
        return v

class SimplexResponse(BaseModel):
    solucion: List[float] = Field(..., description="Solución óptima del problema.")
    valor_optimo: float = Field(..., description="Valor óptimo de la función objetivo.")

class ProblemaSimplex:
    """
    Clase base que define el problema de optimización usando el método Simplex.
    """

    def __init__(self, c, A, b, signos, operacion):
        """
        Inicializa el problema de optimización.

        Args:
            c (list or np.ndarray): Coeficientes de la función objetivo.
            A (list or np.ndarray): Matriz de restricciones.
            b (list or np.ndarray): Vector del lado derecho de las restricciones.
            signos (list): Lista de signos de las restricciones.
            operacion (str): Tipo de operación ('min' o 'max').
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.signos = signos
        self.operacion = operacion
        self.m, self.n = self.A.shape
        self.tableau = self._construir_tableau()

    def _construir_tableau(self):
        """
        Construye el tableau inicial con variables de holgura o exceso según los signos de las restricciones.
        """
        tableau = np.zeros((self.m + 1, self.n + self.m + 1))
        tableau[:self.m, :self.n] = self.A
        tableau[:self.m, -1] = self.b

        for i, signo in enumerate(self.signos):
            if signo == "<=":
                tableau[i, self.n + i] = 1  # Variable de holgura
            elif signo == ">=":
                tableau[i, self.n + i] = -1  # Variable de exceso

        if self.operacion == 'max':
            tableau[-1, :self.n] = -self.c  # Maximización
        else:
            tableau[-1, :self.n] = self.c  # Minimización

        return tableau

    def _obtener_columna_pivote(self):
        """
        Selecciona la columna pivote con el coeficiente más negativo de la fila de la función objetivo.
        """
        ultima_fila = self.tableau[-1, :-1]
        columna_pivote = np.argmin(ultima_fila)
        return columna_pivote if ultima_fila[columna_pivote] < 0 else None

    def _obtener_fila_pivote(self, columna_pivote):
        """
        Obtiene la fila pivote basada en los cocientes positivos.
        """
        rhs = self.tableau[:-1, -1]
        columna = self.tableau[:-1, columna_pivote]
        cocientes = np.where(columna > 0, rhs / columna, np.inf)
        fila_pivote = np.argmin(cocientes)
        return fila_pivote if cocientes[fila_pivote] < np.inf else None

    def _pivotear(self, fila_pivote, columna_pivote):
        """
        Realiza la operación de pivote en el tableau.
        """
        valor_pivote = self.tableau[fila_pivote, columna_pivote]
        self.tableau[fila_pivote, :] /= valor_pivote
        for i in range(self.m + 1):
            if i != fila_pivote:
                self.tableau[i, :] -= self.tableau[i, columna_pivote] * self.tableau[fila_pivote, :]

    def resolver(self):
        """
        Método abstracto para resolver el problema de optimización.
        Debe ser implementado en la clase derivada.
        """
        raise NotImplementedError("Debe implementar el método 'resolver' en la clase derivada")


class SimplexMaximizar(ProblemaSimplex):
    """
    Clase derivada para resolver problemas de maximización usando el método Simplex.
    """

    def __init__(self, c, A, b, signos):
        super().__init__(c, A, b, signos, operacion='max')

    def resolver(self):
        """
        Resuelve el problema de maximización usando el método Simplex.
        """
        while True:
            columna_pivote = self._obtener_columna_pivote()
            if columna_pivote is None:
                break
            fila_pivote = self._obtener_fila_pivote(columna_pivote)
            if fila_pivote is None:
                raise Exception("El problema no tiene solución acotada.")
            self._pivotear(fila_pivote, columna_pivote)

        solucion = np.zeros(self.n)
        for i in range(self.m):
            if np.sum(self.tableau[i, :self.n] == 1) == 1 and np.sum(self.tableau[i, :self.n]) == 1:
                col = np.argmax(self.tableau[i, :self.n])
                solucion[col] = self.tableau[i, -1]

        valor_optimo = self.tableau[-1, -1]
        return solucion.tolist(), valor_optimo


class SimplexMinimizar(ProblemaSimplex):
    """
    Clase derivada para resolver problemas de minimización usando el método Simplex.
    """

    def __init__(self, c, A, b, signos):
        super().__init__(c, A, b, signos, operacion='min')

    def resolver(self):
        """
        Resuelve el problema de minimización usando el método Simplex.
        """
        while True:
            columna_pivote = self._obtener_columna_pivote()
            if columna_pivote is None:
                break
            fila_pivote = self._obtener_fila_pivote(columna_pivote)
            if fila_pivote is None:
                raise Exception("El problema no tiene solución acotada.")
            self._pivotear(fila_pivote, columna_pivote)

        solucion = np.zeros(self.n)
        for i in range(self.m):
            if np.sum(self.tableau[i, :self.n] == 1) == 1 and np.sum(self.tableau[i, :self.n]) == 1:
                col = np.argmax(self.tableau[i, :self.n])
                solucion[col] = self.tableau[i, -1]

        valor_optimo = self.tableau[-1, -1]
        return solucion.tolist(), valor_optimo


@app.post("/simplex", response_model=SimplexResponse)
def resolver_simplex(request: SimplexRequest):
    """
    Resuelve un problema de optimización lineal usando el método Simplex.
    """
    c = request.c
    A = request.A
    b = request.b
    signos = request.signos
    operacion = request.operacion

    # Validaciones adicionales
    if not (len(A) == len(b) == len(signos)):
        raise HTTPException(status_code=400, detail="Las longitudes de A, b y signos deben ser iguales.")

    try:
        if operacion == 'max':
            simplex = SimplexMaximizar(c, A, b, signos)
        else:
            simplex = SimplexMinimizar(c, A, b, signos)

        solucion, valor_optimo = simplex.resolver()
        return SimplexResponse(solucion=solucion, valor_optimo=valor_optimo)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Ejemplo de uso para pruebas (puede eliminarse en producción)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
