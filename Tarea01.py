# Clase RE que representa una expresión regular
from string import ascii_lowercase, ascii_uppercase
from abc import ABC, abstractmethod
from typing import Dict, Set, List, Union,Optional

# Clase Symbol que representa un símbolo ASCII o ε
class Symbol:
    def __init__(self, char: str):
        if char == "ε" or (len(char) == 1 and ord(char) < 128):
            self.char = char
        else:
            raise ValueError("El símbolo debe ser un carácter ASCII o ε")

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.char == other.char

    def __lt__(self, other):
        return isinstance(other, Symbol) and self.char < other.char

    def __str__(self):
        return self.char
    
    # --- Extra ---
    # Add missing function for using the Language Class
    def __hash__(self) -> int:
        return hash(self.char)

# Clase Alphabet que representa un alfabeto
class Alphabet:
    def __init__(self, symbols=None):
        if symbols is None:
            symbols = set()
        self.symbols = set(symbols)

    def add_symbol(self, symbol: Symbol):
        self.symbols.add(symbol)

    def __contains__(self, symbol):
        return symbol in self.symbols

    def __str__(self):
        return "{" + ", ".join(str(s) for s in self.symbols) + "}"

# Clase Word que representa una cadena de símbolos
class Word:
    def __init__(self, symbols=None):
        # --- Extra ---
        # Change from list to tuple to make it Hashable
        if symbols is None:
            symbols = tuple()
        self.symbols = tuple(symbols)

    def __eq__(self, other):
        return isinstance(other, Word) and self.symbols == other.symbols

    def __str__(self):
        if not self.symbols:
            return "ε"
        return "".join(str(symbol) for symbol in self.symbols)
    
    # --- Extra ---
    # Add missing function for using the Language Class
    def __hash__(self) -> int:
        return hash(self.symbols)
    

# Clase Language que representa un lenguaje 
class Language:
    def __init__(self, words=None):
        if words is None:
            words = set()
        self.words = set(words)

    def add_word(self, word: Word):
        self.words.add(word)

    def __contains__(self, word):
        return word in self.words

    def __str__(self):
        return "{" + ", ".join(str(w) for w in self.words) + "}"



# Clase abstracta base para todas las expresiones regulares
class RE(ABC):
    @abstractmethod
    def __str__(self):
        pass

# Clase que representa la expresión regular vacía (Empty)
class Empty(RE):
    def __str__(self):
        return "∅"

# Clase que representa el lenguaje {ε} (Epsilon)
class Epsilon(RE):
    def __str__(self):
        return "ε"

# Clase que representa un símbolo ASCII (Sym)
class Sym(RE):
    def __init__(self, char: str):
        self.char = char

    def __str__(self):
        return self.char

# Clase que representa la concatenación de dos expresiones regulares (Conc)
class Conc(RE):
    def __init__(self, e1: RE, e2: RE):
        self.e1 = e1
        self.e2 = e2

    def __str__(self):
        return f"({self.e1} . {self.e2})"

# Clase que representa la unión de dos expresiones regulares 
class Union(RE):
    def __init__(self, e1: RE, e2: RE):
        self.e1 = e1
        self.e2 = e2

    def __str__(self):
        return f"({self.e1} + {self.e2})"

# Clase que representa la estrella de Kleene de una expresión regular 
class Kleene(RE):
    def __init__(self, e: RE):
        self.e = e

    def __str__(self):
        return f"({self.e})*"


#Ejemplos de uso
r1 = Sym('a')                   # Representa el símbolo 'a'
r2 = Conc(Sym('a'), Sym('b'))   # Representa "a . b"
r3 = Union(Sym('a'), Sym('b'))  # Representa "a + b"
r4 = Kleene(Sym('a'))           # Representa "a*"
r5 = Kleene(r3)                 # Representa "(a + b)*"
#print("Ejemplos de uso de Expresiones Regulares ")
#print(r1)  # Salida: a
#print(r2)  # Salida: (a . b)
#print(r3)  # Salida: (a + b)
#print(r4)  # Salida: (a)*
#print(r5)  # Salida: ((a + b))*

# Clase AutomataFinito como base para AFD y AFNE


# Clase para representar un símbolo o una transición epsilon
class CharWithEpsilon:
    def __init__(self, char: Optional[str] = None):
        self.char = char  # Si char es None, representa una transición epsilon

    def __str__(self):
        return self.char if self.char else 'ε'

    def __eq__(self, other):
        return isinstance(other, CharWithEpsilon) and self.char == other.char

    def __hash__(self):
        return hash(self.char)

# Clase base para autómatas finitos
class AutomataFinito:
    def __init__(self, states=None, init_state=None, final_states=None):
        self.states = states if states is not None else []  # Lista de estados
        self.init_state = init_state if init_state is not None else 0  # Estado inicial
        self.final_states = final_states if final_states is not None else set()  # Conjunto de estados finales
        self.delta = {}  # Diccionario para la función de transición

    def set_states(self, states):
        self.states = states

    def set_init_state(self, init_state):
        self.init_state = init_state

    def set_final_states(self, final_states):
        self.final_states = final_states

# Clase DFA que representa un autómata finito determinista
class DFA(AutomataFinito):
    def __init__(self, states=None, init_state=None, final_states=None):
        super().__init__(states, init_state, final_states)
        self.delta = {}  # Dict[str, Dict[Symbol, str]] para la función de transición

    def set_delta(self, delta: Dict[str, Dict[str, str]]):
        self.delta = delta

    def add_transition(self, from_state: str, symbol: str, to_state: str):
        if from_state not in self.delta:
            self.delta[from_state] = {}
        self.delta[from_state][symbol] = to_state

# Clase NFAE que representa un autómata finito no determinista con transiciones epsilon
class NFAE(AutomataFinito):
    def __init__(self, states=None, init_state=None, final_states=None):
        super().__init__(states, init_state, final_states)
        self.delta = {}  # Dict[str, Dict[CharWithEpsilon, Set[str]]] para la función de transición

    def set_delta(self, delta: Dict[str, Dict[CharWithEpsilon, Set[str]]]):
        self.delta = delta

    def add_transition(self, from_state: str, symbol: CharWithEpsilon, to_states: Set[str]):
        if from_state not in self.delta:
            self.delta[from_state] = {}

        # --- Extra ---
        # Cast to Str para no tener discordancia en los symbolos por almacenamiento en memoria
        if str(symbol) not in self.delta[from_state]:
            self.delta[from_state][str(symbol)] = set()

        # --- Extra ---
        # Agregar el propio estado cuando el symbolo es ε
        if str(symbol) == 'ε':
             self.delta[from_state][str(symbol)].add(from_state)

        # --- Extra ---
        # Para permitir varias transiciones del mismo simbolo en el mismo estado
        self.delta[from_state][str(symbol)].update(
            self.delta[from_state][str(symbol)].union(to_states)
        )

    def has_epsilon_transitions(self, state: str) -> bool:
        """Verifica si el estado actual puede transicionar con epsilon."""
        return str(CharWithEpsilon(None)) in self.delta.get(state, {})
    
    # --- Extra ---
    # Agregado para imprimir las pruebas
    def __str__(self) -> str:
        return str(self.delta)

# Ejemplo de uso
nfae = NFAE(states=['q0', 'q1', 'q2'], init_state='q0', final_states={'q2'})
epsilon = CharWithEpsilon()  # Epsilon
nfae.add_transition('q0', epsilon, {'q1'})
nfae.add_transition('q1', CharWithEpsilon('a'), {'q2'})
nfae.add_transition('q0', epsilon, {'q2'})
nfae.add_transition('q2', epsilon, {'q1'})

#print(nfae.has_epsilon_transitions('q0'))  # Salida: True
#print(nfae.has_epsilon_transitions('q1'))  # Salida: False

# IMPLEMENTA LAS FUNCIONES QUE SE SOLICITAN EN EL PDF

# --- Simbolos y Cadenas ---

def first(string: Word) -> Symbol:
    """
        Funcion que regresa el simbolo mas a la derecha de una cadena

        Parameters
        ----------
        string: Word
            La cadena
        
        Returns
        -------
        symbol: Symbol
            El simbolo mas a la derecha
    """

    # Caso base
    if len(string.symbols) == 0:
        return Symbol("ε")
    # El simbolo mas a la derecha (i.e El ultimo)
    else:
        return string.symbols[-1]

def prefix(string: Word) -> Word:
    """
    Funcion que regresa toda la cadena, menos el simbolo mas a la derecha

        Parameters
        ----------
        string: Word
            La cadena
        
        Returns
        -------
        word: Word
            La cadena sin el simbolo mas a la derecha
    """


    # Caso Base
    if len(string.symbols) == 0:
        return Word([Symbol("ε")])
    # Caso con 1 simbolo
    elif len(string.symbols)==1:
        return Word([string.symbols[0]])
    # Caso con >1 simbolos
    else:
        return Word(string.symbols[:-1])
 
    
def membership(string: Word, l: Language) -> bool:
    """
    Funcion que determina si una cadena pertenece a un Lenguaje

        Parameters
        ----------
        string: Word
            La cadena
        l: Language
            El lenguaje
        
        Returns
        -------
        bool: bool
            True o False si la cadena pertenece al lenguaje
    """
    # Podemos usar esto gracias a las funciones
    # __eq__ de Word y __contains__ de Language
    return string in l

# --- ε-AFNs ---

def closure(E:NFAE, s:str) -> set:
    """
    Funcion que determina la clausara epsilon de un estado en un ε-AFN

        Parameters
        ----------
        E: NFAE
            El ε-AFN
        s: str
            Un estado Especifico
        
        Returns
        -------
        cl_epsilon: set
            La clausura epsilon del estado
    """

    cl_epsilon = [s]

    for i in cl_epsilon: # Recorre todo los estados posibles
        if E.has_epsilon_transitions(i): # Determina si existe una transicion epsilon
            # Obtiene todo los estados posibles que se pueden llegar con epsilon
            transitions = E.delta[i][str(CharWithEpsilon())] 
            for transition in transitions: # Recorrer todos los estado alcanzables con ε
                if transition not in cl_epsilon:
                    # Si no estaba el estado, se agrega
                    cl_epsilon.append(transition)
   
    return set(cl_epsilon)

def accept(E: NFAE, string:Word) -> bool:
    """
    Funcion que determina si una cadena es aceptana en un ε-AFN

        Parameters
        ----------
        E: NFAE
            El ε-AFN
        string: Word
            La cadena
        
        Returns
        -------
        bool: bool
            True o False
    """

    def extended_closure(E1: NFAE, states:set) -> set:
        """
        Funcion Auxiliar
        Extiende la definicion de Cerradura epsilon

            Parameters
            ----------
            E1: NFAE
                El ε-AFN
            states: set
                Estados
            
            Returns
            -------
            cl_epsilon: set
                La cerradura epsilon de todos los estados de states
        """
        
        cl_epsilon = set()
        for state in states:
            cl_epsilon = cl_epsilon.union(closure(E1, state))
        
        return cl_epsilon
    
    def extended_delta(E2: NFAE, string2:Word, states:set) -> set:
        """
        Funcion Auxiliar
        Extiende la definicion de funcion de transicion

            Parameters
            ----------
            E2: NFAE
                El ε-AFN
            string2: Word
                Cadena a la que se le aplica la funcion de transicion
            states: set
                Estados a los que se les busca la transicion
            
            Returns
            -------
            reachable_states: set
                Conjunto de estados alcanzables
        """

        # Cerradura extendida para los estados
        current_states = extended_closure(E2, states)
        
        # Caso base
        # delta(q,ε) = Cl_ε(q)
        if  not string2.symbols:
            return current_states
        
        reachable_states = set()

        # Buscamos todos los estados alcanzables con el primer simbolo
        for state in current_states:
            if str(string2.symbols[0]) in E2.delta.get(state, {}):
                # Union de los estados alcanzables
                reachable_states = reachable_states.union(E2.delta[state][str(string2.symbols[0])])

        # Cerradura epsilon de los estados alcanzables
        reachable_states = extended_closure(E2, reachable_states)

        # Definicion recursiva con el siguiente simbolo 
        # y los estados alcanzables
        return extended_delta(E2, Word(string2.symbols[1:]), reachable_states)

    initial_state = E.init_state
    reachable_states = extended_delta(E, string, {initial_state})

    # Definicion de aceptacion
    return True if reachable_states.intersection(E.final_states) else False


def empty() -> NFAE:
    """
    Funcion que regresa el automata que acepta el lenguaje vacio

        Parameters
        ----------
        
        Returns
        -------
        emptyEAFN: NFAE
            El automata
    """
    emptyEAFN = NFAE(
        states = ["q0"],
        init_state="q0",
        final_states={} # Sin estados finales
    )
    return emptyEAFN

def anf_epsilon() -> NFAE:
    """
    Funcion que regresa el automata que acepta la cadena epsilon

        Parameters
        ----------
        
        Returns
        -------
        epsilonEAFN: NFAE
            El automata
    """
    epsilonEAFN = NFAE(
        states = ["q0"],
        init_state="q0",
        final_states={"q0"} # Estado inicial el mismo estado final
    )
    return epsilonEAFN

def symbol(a:CharWithEpsilon) -> NFAE:
    """
    Funcion que regresa el automata que acepta la cadena a

        Parameters
        ----------
        a: CharWithEpsilon
            El simbolo a aceptar
        
        Returns
        -------
        aNFAE: NFAE
            El automata
    """

    aNFAE = NFAE(
        states = ["q0", "q1"],
        init_state="q0",
        final_states={"q1"}
    )
    # Una transicion del inicial al final por medio de "a"
    aNFAE.add_transition("q0", a, {"q1"})
    return aNFAE

def union(E1: NFAE, E2: NFAE) -> NFAE:
    """
    Funcion que regresa el automata que acepta la Union de dos Automatas

        Parameters
        ----------
        E1: NFAE
            Automata 1
        
        E2: NFAE
            Automata 2
        
        Returns
        -------
        unionE: NFAE
            El automata union
    """

    # Renombramos los estados
    E1_states = [state + "_e1" for state in E1.states]
    E2_states = [state + "_e2" for state in E2.states]

    E1_final_states = {state + "_e1" for state in E1.final_states}
    E2_final_states = {state + "_e2" for state in E2.final_states}

    E1_init_state = E1.init_state + "_e1"
    E2_init_state = E2.init_state + "_e2"

    unionE = NFAE(
        states= E1_states + E2_states + ["q0_union"],
        init_state="q0_union",
        final_states=E1_final_states.union(E2_final_states)
    )

    # Agregamos la transiciones epsilon a cada automata 
    unionE.add_transition("q0_union", CharWithEpsilon(), {E1_init_state})
    unionE.add_transition("q0_union", CharWithEpsilon(), {E2_init_state})

    # Agregamos las transiciones de los automatas
    for state in E1.states:
        for transition in E1.delta.get(state,{}).keys():
            reachable_states_E1 = {next_state + "_e1" for next_state in E1.delta[state][transition]}
            unionE.add_transition(state + "_e1", CharWithEpsilon(transition), reachable_states_E1)

    for state in E2.states:
        for transition in E2.delta.get(state,{}).keys():
            reachable_states_E2 = {next_state + "_e2" for next_state in E2.delta[state][transition]}
            unionE.add_transition(state + "_e2", CharWithEpsilon(transition), reachable_states_E2)

    return unionE

def concat(E1: NFAE, E2: NFAE) -> NFAE:
    """
    Funcion que regresa el automata que acepta la Concatenacion de dos Automatas

        Parameters
        ----------
        E1: NFAE
            Automata 1
        
        E2: NFAE
            Automata 2
        
        Returns
        -------
        concatE: NFAE
            El automata union
    """

    # Renombramos los estados
    E1_states = [state + "_e1" for state in E1.states]
    E2_states = [state + "_e2" for state in E2.states]

    E1_final_states = {state + "_e1" for state in E1.final_states}
    E2_final_states = {state + "_e2" for state in E2.final_states}

    E1_init_state = E1.init_state + "_e1"
    E2_init_state = E2.init_state + "_e2"

    concatE = NFAE(
        states= E1_states + E2_states,
        init_state=E1_init_state,
        final_states=E2_final_states
    )

    # Agregamos las transiciones del primer automata
    for state in E1.states:
        for transition in E1.delta.get(state,{}).keys():
            reachable_states_E1 = {next_state + "_e1" for next_state in E1.delta[state][transition]}
            concatE.add_transition(state + "_e1", CharWithEpsilon(transition), reachable_states_E1)

    # Agremos la transicion epsilon al segundo automata
    for final_state in E1_final_states:
        concatE.add_transition(final_state, CharWithEpsilon(), {E2_init_state})

    # Agregamos las transiciones del segundo automata
    for state in E2.states:
        for transition in E2.delta.get(state,{}).keys():
            reachable_states_E2 = {next_state + "_e2" for next_state in E2.delta[state][transition]}
            concatE.add_transition(state + "_e2", CharWithEpsilon(transition), reachable_states_E2)

    return concatE

def kleene(E:NFAE) -> NFAE:
    """
    Funcion que regresa el automata que acepta la estrella de Kleene de un Automata

        Parameters
        ----------
        E: NFAE
            Automata
        
        Returns
        -------
        kleeneE: NFAE
            El automata union
    """

    # Renombramos los estados
    E_states = [state + "_e" for state in E.states]
    E_final_states = {state + "_e" for state in E.final_states}
    E_init_state = E.init_state + "_e"

    kleeneE = NFAE(
        states= E_states,
        init_state=E_init_state,
        final_states={E_init_state}.union(E_final_states)
    )

    # Agregamos las transiciones del Automata
    for state in E.states:
        for transition in E.delta.get(state,{}).keys():
            reachable_states_E1 = {next_state + "_e" for next_state in E.delta[state][transition]}
            kleeneE.add_transition(state + "_e", CharWithEpsilon(transition), reachable_states_E1)

    # Agregamos las transiciones epsilon del estado final al inicial
    for final_state in E_final_states:
        kleeneE.add_transition(final_state, CharWithEpsilon(), {E_init_state})
    
    return kleeneE

# --- Expresiones Regulares ---

def toEAFN(e:RE) -> NFAE:
    """
    Funcion que regresa el automata que acepta la expresion regular e

        Parameters
        ----------
        e: RE
            La expersion regular
        
        Returns
        -------
        : NFAE
            El automata a regresar
    """
    if isinstance(e, Sym):
        return symbol(CharWithEpsilon(str(e)))
    elif isinstance(e, Epsilon):
        return anf_epsilon()
    elif isinstance(e, Conc):
        return concat(toEAFN(e.e1), toEAFN(e.e2))
    elif isinstance(e, Union):
        return union(toEAFN(e.e1), toEAFN(e.e2))
    elif isinstance(e, Kleene):
        return kleene(toEAFN(e.e))
    else:
        return empty()
    
# --- Correo Electronico ---

def verify(mail:str) -> bool:
    """
    Funcion que verifica si una cadena de mail es correcta

        Parameters
        ----------
        mail: str
            La cadena de mail
        
        Returns
        -------
        : bool
            True o False
    """

    # Formamos el Alfabeto
    all_numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    all_letters = (
        list(ascii_lowercase)
        + list(ascii_uppercase)
        + all_numbers
    ) 
    ascii_re = Sym(all_letters[0])

    for char in all_letters[1:]:
        ascii_re = Union(ascii_re, Sym(char))

    # La expersion regular que determina si es aceptada o no
    mailRE = Conc(
        Conc(
            Conc(
                    Conc(
                        ascii_re,
                        Kleene(ascii_re)
                    ), # Nombre del Email
                    Sym("@") # Arroba
                ),
            Conc(
                ascii_re,
                Kleene(ascii_re)
            ) # Dominio asociado al email
        ),
        Conc(
            Conc(
                Sym("."),
                Conc(
                    ascii_re,
                    Kleene(ascii_re)
                )
            ), 
            Kleene(
                    Conc(
                        Sym("."),
                        Conc(
                            ascii_re,
                            Kleene(ascii_re)
                        )
                )
            ) # .com, .net , etc.
        )
    )

    # Creamos el automata que acepta el lenguaje
    # Generado por la expresion regular
    mailEAFN = toEAFN(mailRE)

    # Convertimos la cadena de Mail a typo Word
    wordSym = []
    for char in mail:
        wordSym.append(CharWithEpsilon(char))

    mailString = Word(wordSym)

    # Determinamos si se acepta o no
    return accept(mailEAFN, mailString)
    
if __name__ == "__main__":
    # Funcion Main para testar el funcionamiento del codigo

    test_response = open("test_response.txt", "w")

    def testF(
        test,
        expect,
        f,
        **kwargs
    ):  
        # Funcion Aux para ejecutar pruebas

        print(f"Funcion: {f.__name__}", file=test_response)
        counter = 0
        
        for toTest, expected in zip(test, expect):
            print(f"Test: {toTest}, Expected: {expected}", file=test_response)
            try:
                assert f(toTest, **kwargs) == expected
            except:
                print(f"Failed: {toTest}, Result: {f(toTest, **kwargs)}", file=test_response)
                counter+=1
        
        print(f"\nTests aprobados: {100*(1-counter/len(test))}% \n\n", file=test_response)

    # Exteded Test
    def extended_testF(
        test,
        expect,
        strings,
        f,
        **kwargs
    ):  
        # Funcion Aux para ejecutar pruebas

        print(f"Funcion: {f.__name__}" ,file=test_response)
        counter = 0
        
        for toTest, expected, cadena in zip(test, expect, strings):
            print(f"Test: [{toTest}, {cadena}], Expected: {expected}", file=test_response)
            try:
                assert f(toTest, cadena, **kwargs) == expected
            except:
                print(f"Failed: [{toTest}, {cadena}], Result: {f(toTest, cadena, **kwargs)}", file=test_response)
                counter+=1
        
        print(f"\nTests aprobados: {100*(1-counter/len(test))}% \n\n", file=test_response)

        

    # --- Simbolos y Cadenas ---

    # Simbolos iniciales
    sym0 = Symbol("0")
    sym1 = Symbol("1")

    # Ejemplo de cadenas
    cadena1 = Word([sym0]) # 0
    cadena2 = Word([sym1]) # 1
    cadena3 = Word([sym0, sym1]) # 01
    cadena4 = Word([sym0, sym1, sym0]) # 010

    # Ejemplo de lenguaje
    language1 = Language(
            words=[
                cadena1,
                cadena2,
                cadena3
            ]
        )

    # First
    testF(
        [cadena1, cadena2, cadena3], 
        [sym0, sym1, sym1],
        first
    )

    # Prefix
    testF(
        [cadena2, cadena3, cadena4], 
        [Word([sym1]), Word([sym0]), Word([sym0, sym1])],
        prefix
    )

    # Membership
    testF(
        [cadena1, cadena2, cadena4], 
        [True, True, False],
        membership,
        l=language1
    )

    # --- ε-AFNs ---

    # Alfabeto
    char0 = CharWithEpsilon('0')
    char1 = CharWithEpsilon('1')
    epsilon = CharWithEpsilon()

    # Automata ejemplo
    nfae1 = NFAE(states=['q0', 'q1', 'q2'], init_state="q0", final_states={'q1'})

    # Transiciones
    nfae1.add_transition('q0', char0, {'q1'})
    nfae1.add_transition('q0', epsilon, {'q2'})
    nfae1.add_transition('q0', char1, {'q0'})
    nfae1.add_transition('q1', char0, {'q1'})
    nfae1.add_transition('q1', char1, {'q0'})

    # NFAE que acepta cadenas que terminan en 0
    nfae2 = NFAE(states=['q0', 'q1'], init_state="q0", final_states={'q1'})

    # Transiciones
    nfae2.add_transition('q0', char0, {'q1'})
    nfae2.add_transition('q0', char1, {'q0'})
    nfae2.add_transition('q1', char0, {'q1'})
    nfae2.add_transition('q1', char1, {'q0'})

    # NFAE que acepta cadenas que terminan en 1
    nfae3 = NFAE(states=['q0', 'q1'], init_state="q0", final_states={'q1'})

    # Transiciones
    nfae3.add_transition('q0', char0, {'q0'})
    nfae3.add_transition('q0', char1, {'q0'})
    nfae3.add_transition('q0', char1, {'q1'})
    nfae3.add_transition('q1', char0, {'q0'})
    nfae3.add_transition('q1', char1, {'q1'})

    # Automata que solo acepta 01
    nfae4 = NFAE(states=['q0', 'q1', 'q2', 'q3'], init_state="q0", final_states={'q2'})

    # Transiciones
    nfae4.add_transition('q0', char0, {'q1'})
    nfae4.add_transition('q0', char1, {'q3'})
    nfae4.add_transition('q1', char0, {'q3'})
    nfae4.add_transition('q1', char1, {'q2'})
    nfae4.add_transition('q2', char0, {'q3'})
    nfae4.add_transition('q2', char1, {'q3'})
    nfae4.add_transition('q3', char0, {'q3'})
    nfae4.add_transition('q3', char1, {'q3'})

    # Closure
    testF(
        [nfae1],
        [{"q0", "q2"}],
        closure,
        s = "q0"
    )

    # Accept
    extended_testF(
        [nfae2, nfae2, nfae3, nfae3],
        [True, False, True, False],
        [
            Word([char0,char0,char0]),
            Word([char0,char1]),
            Word([char0,char1]),
            Word([char0,char1,char0]),
        ],
        accept
    )

    # Empty
    print("Prueba de Empty" ,file=test_response)
    extended_testF(
        [empty(), empty()],
        [False, False],
        [
            Word([char0,char0,char0]),
            Word([char0,char1])
        ],
        accept
    )

    # Epsilon
    print("Prueba de Epsilon", file=test_response)
    extended_testF(
        [anf_epsilon(), anf_epsilon()],
        [False, True],
        [
            Word([char0,char0,char0]),
            Word()
        ],
        accept
    )

    # Symbol
    print("Prueba de Symbol", file=test_response)
    extended_testF(
        [symbol(char0), symbol(char1), symbol(char1)],
        [False, False, True],
        [
            Word([char0,char0]),
            Word([char0,char1]),
            Word([char1])
        ],
        accept
    )

    # Union
    print("Prueba de Union", file=test_response)
    extended_testF(
        [union(nfae2,nfae3), union(nfae2,nfae3), union(nfae2,nfae3)],
        [True, True, False],
        [
            Word([char0,char0]),
            Word([char0,char1]),
            Word()
        ],
        accept
    )

    # Union
    print("Prueba de Concat", file=test_response)
    extended_testF(
        [concat(nfae2,nfae3), concat(nfae2,nfae3), concat(nfae2,nfae3)],
        [False, True, False],
        [
            Word([char0,char0]),
            Word([char0,char1]),
            Word()
        ],
        accept
    )

    # Kleene
    print("Prueba de Kleene", file=test_response)
    extended_testF(
        [kleene(nfae4), kleene(nfae4), kleene(nfae4)],
        [False, True, True],
        [
            Word([char0,char0]),
            Word([char0,char1,char0,char1]),
            Word()
        ],
        accept
    )


    # --- Expresiones Regulares ---

    # Alfabeto
    chara = CharWithEpsilon('a')
    charb = CharWithEpsilon('b')

    # Expersiones Regulares
    r2 = Conc(Sym('a'), Sym('b'))   # Representa "a . b"
    r3 = Union(Sym('a'), Sym('b'))  # Representa "a + b"
    r4 = Kleene(Sym('a'))           # Representa "a*"
    r5 = Kleene(r3)                 # Representa "(a + b)*"
    r6 = Union(r2,Sym('a'))         # Representa "((a . b) + a)"
    r7 = Union(r2,r3)               # Representa "((a . b) + (a + b))"

    # Get AFNs
    nfae_r2 = toEAFN(r2)
    nfae_r3 = toEAFN(r3)
    nfae_r4 = toEAFN(r4)
    nfae_r5 = toEAFN(r5)
    nfae_r6 = toEAFN(r6)
    nfae_r7 = toEAFN(r7)

    # Prueba toEAFN
    print("Prueba toEAFN", file=test_response)
    extended_testF(
        [nfae_r2, nfae_r3, nfae_r4, nfae_r5, nfae_r6, nfae_r7, nfae_r7, nfae_r2],
        [True, True, True, True, True, True, True, False],
        [
            Word([chara, charb]),
            Word([charb]),
            Word([chara, chara]),
            Word([charb,chara, charb]),
            Word([chara]),
            Word([chara]),
            Word([chara, charb]),
            Word([chara, charb, charb])
        ],
        accept
    )

    # --- Correo Electronico ---

    # Ejemplos de mails
    mail0 = "prueba@mail.com"
    mail1 = "prueba2@mail.com.mx"
    mail2 = "a@ab"
    mail3 = "@.a"
    mail4 = "a@"
    mail5 = "a.a"

    # Verify
    testF(
        [mail0, mail1, mail2, mail3, mail4, mail5],
        [True, True, False, False, False],
        verify
    )

    # Close the file to write
    test_response.close()

    # Print the response from the text file
    read_response = open("test_response.txt", "r")
    print(read_response.read())

