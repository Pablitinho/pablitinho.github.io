[Translated Content]
# Módulo 5: Arquitecturas Eficientes

## Lección 1: MobileNet

### Introducción a MobileNet

MobileNet representa un punto de inflexión en el diseño de redes neuronales convolucionales, marcando el inicio de una nueva era de arquitecturas específicamente optimizadas para dispositivos con recursos limitados. Desarrollada por investigadores de Google, esta familia de arquitecturas aborda directamente uno de los mayores desafíos en la implementación práctica de CNN: cómo desplegar modelos de alta precisión en dispositivos móviles y embebidos con severas restricciones de memoria, potencia de cálculo y energía.

A diferencia de arquitecturas anteriores como ResNet o DenseNet, que priorizaban la precisión sobre la eficiencia, MobileNet introduce un nuevo paradigma donde la eficiencia computacional es un objetivo de diseño primario, no una consideración secundaria. Este cambio de enfoque refleja la creciente necesidad de ejecutar modelos de aprendizaje profundo directamente en dispositivos de usuario final, sin depender de servidores en la nube, para aplicaciones que requieren baja latencia, funcionamiento sin conexión o privacidad de datos.

La innovación central de MobileNet es la sustitución de las convoluciones estándar por convoluciones separables en profundidad (depthwise separable convolutions), una factorización que reduce drásticamente el número de operaciones y parámetros mientras preserva gran parte de la capacidad representacional. Además, MobileNet introduce hiperparámetros específicos para ajustar el equilibrio entre latencia y precisión, permitiendo adaptar la arquitectura a diferentes escenarios de despliegue.

Desde su introducción en 2017, MobileNet ha evolucionado a través de múltiples versiones (MobileNetV1, V2, V3), cada una incorporando nuevas técnicas para mejorar la eficiencia y precisión. Su impacto trasciende su aplicación directa, habiendo influido significativamente en el diseño de arquitecturas eficientes posteriores y establecido nuevos estándares para la evaluación de modelos en términos de precisión por operación.

### Convoluciones Separables en Profundidad

La innovación fundamental que define a MobileNet es el uso sistemático de convoluciones separables en profundidad en lugar de convoluciones estándar.

#### Convolución Estándar vs. Separable en Profundidad

Una convolución estándar realiza simultáneamente el filtrado espacial y la combinación de canales. Para una capa con M canales de entrada, N canales de salida, y filtros de tamaño K×K, el costo computacional es proporcional a:

M × N × K × K × H × W

Donde H×W es la resolución espacial del mapa de características.

La convolución separable en profundidad factoriza esta operación en dos pasos:

1. **Convolución en Profundidad (Depthwise Convolution)**: Aplica un filtro espacial K×K a cada canal de entrada por separado, sin combinar canales.
   - Costo: M × K × K × H × W

2. **Convolución Puntual (Pointwise Convolution)**: Aplica filtros 1×1 para combinar los canales filtrados y crear nuevos mapas de características.
   - Costo: M × N × H × W

El costo total es proporcional a:
M × K × K × H × W + M × N × H × W = M × H × W × (K² + N)

#### Reducción de Costo Computacional

Comparando con la convolución estándar, la reducción en costo computacional es:

(M × K × K × H × W + M × N × H × W) / (M × N × K × K × H × W) = (K² + N) / (N × K²)

Para valores típicos (K=3, N=256), esto representa una reducción de aproximadamente 8-9 veces en el número de operaciones.

Esta factorización separa explícitamente dos responsabilidades:
- La convolución en profundidad se encarga del filtrado espacial
- La convolución puntual se encarga de la combinación de canales

#### Implementación Práctica

En la implementación de MobileNet, cada bloque separable en profundidad sigue esta secuencia:
1. Convolución en profundidad con un filtro por canal
2. Normalización por lotes (Batch Normalization)
3. Activación ReLU
4. Convolución puntual (1×1) para combinar canales
5. Normalización por lotes
6. Activación ReLU

Esta estructura se repite a lo largo de la red, con reducciones ocasionales de resolución espacial mediante stride=2 en las convoluciones en profundidad.

### Hiperparámetros: Multiplicador de Anchura y Resolución

Una característica distintiva de MobileNet es la introducción de dos hiperparámetros específicos para ajustar el equilibrio entre precisión y eficiencia:

#### Multiplicador de Anchura (α)

El multiplicador de anchura (α) controla uniformemente el número de canales en cada capa:
- Si α = 1, se utiliza el número completo de canales (modelo base)
- Si α < 1, se reduce proporcionalmente el número de canales en todas las capas

Por ejemplo, con α = 0.5, cada capa tendrá la mitad de canales, reduciendo el costo computacional aproximadamente por un factor de 4 (ya que tanto las convoluciones en profundidad como las puntuales escalan con el número de canales).

Valores típicos: α ∈ {1, 0.75, 0.5, 0.25}

#### Multiplicador de Resolución (ρ)

El multiplicador de resolución (ρ) controla la resolución de entrada y, consecuentemente, la resolución de todos los mapas de características internos:
- Si ρ = 1, se utiliza la resolución completa (típicamente 224×224)
- Si ρ < 1, se reduce proporcionalmente la resolución

Por ejemplo, con ρ = 0.5, la entrada sería 112×112, reduciendo el costo computacional aproximadamente por un factor de 4 (ya que el costo escala cuadráticamente con la resolución).

Valores típicos: ρ ∈ {1, 0.857, 0.714, 0.571}

#### Impacto en Precisión y Eficiencia

Estos hiperparámetros permiten generar una familia de modelos con diferentes equilibrios entre precisión y eficiencia:

- Reducir α disminuye el número de parámetros y operaciones, pero también la capacidad representacional
- Reducir ρ disminuye el costo computacional cuadráticamente, pero puede perder información espacial fina

Los autores de MobileNet demostraron empíricamente que reducir la anchura (α) generalmente tiene un impacto menor en la precisión que reducir la resolución (ρ) para un mismo nivel de reducción computacional.

### MobileNetV1 vs MobileNetV2

La arquitectura MobileNet ha evolucionado significativamente desde su introducción, con MobileNetV2 incorporando importantes mejoras sobre el diseño original.

#### Arquitectura MobileNetV1

MobileNetV1, introducida en 2017, estableció el paradigma básico:

1. **Estructura General**:
   - Convolución estándar inicial 3×3 con stride=2
   - 13 bloques de convolución separable en profundidad
   - Average Pooling global
   - Capa completamente conectada para clasificación

2. **Bloque Básico**:
   - Convolución en profundidad 3×3
   - Batch Normalization + ReLU
   - Convolución puntual 1×1
   - Batch Normalization + ReLU

3. **Características**:
   - 4.2 millones de parámetros (α=1)
   - 569 millones de operaciones de multiplicación-acumulación
   - 70.6% de precisión top-1 en ImageNet

#### Innovaciones en MobileNetV2

MobileNetV2, presentada en 2018, introdujo dos innovaciones clave:

1. **Conexiones Residuales Lineales**:
   - Similar a ResNet, pero con una diferencia crucial: la última activación ReLU se elimina
   - Esto preserva información en el espacio de características de baja dimensionalidad

2. **Bloque de Cuello de Botella Invertido (Inverted Bottleneck)**:
   - A diferencia del diseño bottleneck tradicional que reduce y luego expande dimensiones, MobileNetV2 primero expande y luego reduce
   - Secuencia: expansión mediante convolución 1×1, filtrado mediante convolución en profundidad, proyección mediante convolución 1×1 sin activación

3. **Estructura del Bloque**:
   - Convolución puntual 1×1 para expansión (típicamente 6x)
   - Batch Normalization + ReLU6
   - Convolución en profundidad 3×3
   - Batch Normalization + ReLU6
   - Convolución puntual 1×1 para reducción (sin activación)
   - Conexión residual (si las dimensiones coinciden)

#### Comparativa de Rendimiento

MobileNetV2 logró mejoras significativas sobre V1:

| Modelo | Parámetros | Operaciones | Precisión Top-1 (ImageNet) |
|--------|------------|-------------|----------------------------|
| MobileNetV1 (α=1) | 4.2M | 569M | 70.6% |
| MobileNetV2 (α=1) | 3.4M | 300M | 72.0% |

Las mejoras clave incluyen:
- Menor número de parámetros (-19%)
- Menor costo computacional (-47%)
- Mayor precisión (+1.4%)

Estas mejoras demuestran la efectividad del diseño de cuello de botella invertido y las conexiones residuales lineales.

### MobileNetV3: Búsqueda de Arquitectura Neural

MobileNetV3, introducida en 2019, representa un salto cualitativo en el diseño de arquitecturas eficientes al incorporar técnicas de búsqueda de arquitectura neural (NAS) y optimizaciones específicas para hardware.

#### Innovaciones Principales

1. **Búsqueda de Arquitectura Neural (NAS)**:
   - Utiliza algoritmos automatizados para optimizar la estructura de la red
   - Combina MnasNet (búsqueda de bloques) con NetAdapt (optimización por capa)
   - Optimiza directamente para objetivos específicos de latencia en hardware real

2. **Nuevo Bloque de Construcción**:
   - Incorpora el "Squeeze-and-Excitation" (SE) module para atención de canal
   - Utiliza la función de activación "hard-swish" (h-swish) más eficiente
   - Rediseña la primera y última capa para reducir latencia

3. **Variantes Optimizadas**:
   - **MobileNetV3-Large**: Optimizada para máxima precisión dentro de restricciones de latencia
   - **MobileNetV3-Small**: Optimizada para casos de uso con recursos extremadamente limitados

#### Mejoras de Rendimiento

MobileNetV3 logró mejoras significativas sobre sus predecesores:

| Modelo | Parámetros | Operaciones | Precisión Top-1 (ImageNet) | Latencia Relativa |
|--------|------------|-------------|----------------------------|-------------------|
| MobileNetV2 | 3.4M | 300M | 72.0% | 1.0x |
| MobileNetV3-Large | 5.4M | 219M | 75.2% | 0.98x |
| MobileNetV3-Small | 2.5M | 66M | 67.4% | 0.45x |

MobileNetV3-Large logra:
- +3.2% mayor precisión que MobileNetV2
- 27% menos operaciones
- Latencia ligeramente menor en dispositivos móviles

MobileNetV3-Small ofrece:
- Rendimiento competitivo con 78% menos operaciones que MobileNetV2
- Menos de la mitad de latencia

### Aplicaciones en Dispositivos Móviles

MobileNet ha encontrado amplia aplicación en dispositivos móviles y embebidos, donde las restricciones de recursos son significativas.

#### Casos de Uso Principales

1. **Visión en Dispositivo**:
   - Clasificación de imágenes en tiempo real
   - Detección de objetos (mediante SSDLite, una versión eficiente de SSD)
   - Segmentación semántica (mediante adaptaciones eficientes de DeepLabv3)
   - Reconocimiento facial y de gestos

2. **Realidad Aumentada**:
   - Detección y seguimiento de objetos
   - Estimación de pose
   - Segmentación para efectos visuales

3. **Asistentes Inteligentes**:
   - Reconocimiento visual para asistentes de voz
   - Análisis de contexto visual

4. **Fotografía Computacional**:
   - Mejora de imágenes
   - Bokeh artificial (desenfoque de fondo)
   - Clasificación de escenas para optimización de parámetros

#### Ventajas en Entornos Móviles

MobileNet ofrece varias ventajas críticas para aplicaciones móviles:

1. **Baja Latencia**: Procesamiento en tiempo real con tiempos de respuesta típicos de 10-30ms en dispositivos de gama media-alta.

2. **Funcionamiento Sin Conexión**: No requiere conectividad a servidores, permitiendo uso en áreas sin cobertura o con conectividad limitada.

3. **Privacidad**: Los datos se procesan localmente sin necesidad de enviarlos a servidores externos.

4. **Ahorro de Batería**: Menor consumo energético comparado con el envío constante de datos a servidores.

5. **Adaptabilidad**: Los hiperparámetros permiten ajustar el modelo según las capacidades específicas del dispositivo.

### Implementación Simplificada de MobileNetV2

A continuación, se presenta una implementación conceptual simplificada de un bloque de MobileNetV2 utilizando TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras import layers

def inverted_residual_block(x, filters, stride, expansion_factor, use_residual=True):
    """
    Implementación de un bloque de cuello de botella invertido de MobileNetV2
    
    Args:
        x: Tensor de entrada
        filters: Número de filtros de salida
        stride: Stride para la convolución en profundidad
        expansion_factor: Factor de expansión para la primera convolución 1x1
        use_residual: Si es True, añade una conexión residual
    
    Returns:
        Tensor de salida del bloque
    """
    # Guardar entrada para la conexión residual
    shortcut = x
    input_filters = x.shape[-1]
    
    # Fase de expansión
    x = layers.Conv2D(input_filters * expansion_factor, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)  # ReLU6: min(max(0, x), 6)
    
    # Fase de filtrado espacial (convolución en profundidad)
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    # Fase de proyección
    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    # No hay activación después de la proyección
    
    # Añadir conexión residual si es posible
    if use_residual and stride == 1 and input_filters == filters:
        x = layers.Add()([shortcut, x])
    
    return x

def create_mobilenetv2(input_shape=(224, 224, 3), num_classes=1000, alpha=1.0):
    """
    Crea un modelo MobileNetV2
    
    Args:
        input_shape: Forma del tensor de entrada
        num_classes: Número de clases para la clasificación
        alpha: Multiplicador de anchura
    
    Returns:
        Modelo MobileNetV2
    """
    inputs = layers.Input(shape=input_shape)
    
    # Primera capa convolucional
    x = layers.Conv2D(int(32 * alpha), 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    # Configuración de bloques: (expansion_factor, filters, num_blocks, stride)
    block_settings = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1)
    ]
    
    # Bloques de cuello de botella invertido
    for expansion, filters, num_blocks, stride in block_settings:
        # Ajustar número de filtros según alpha
        filters = int(filters * alpha)
        
        # Primer bloque con stride especificado
        x = inverted_residual_block(x, filters, stride, expansion)
        
        # Bloques restantes con stride=1
        for _ in range(1, num_blocks):
            x = inverted_residual_block(x, filters, 1, expansion)
    
    # Última capa convolucional
    x = layers.Conv2D(int(1280 * alpha), 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    # Clasificador
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Crear MobileNetV2 con alpha=1.0
mobilenetv2 = create_mobilenetv2(alpha=1.0)

# Compilar el modelo
mobilenetv2.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Resumen del modelo
print("MobileNetV2 Summary:")
mobilenetv2.summary()
```

Esta implementación simplificada captura los elementos esenciales de MobileNetV2, incluyendo los bloques de cuello de botella invertido y las conexiones residuales lineales.

### Ventajas y Limitaciones

#### Ventajas de MobileNet

1. **Eficiencia Computacional**: Drástica reducción en número de operaciones y parámetros comparado con arquitecturas tradicionales.

2. **Latencia Reducida**: Tiempos de inferencia significativamente menores, cruciales para aplicaciones en tiempo real.

3. **Tamaño Compacto**: Modelos más pequeños que requieren menos memoria y almacenamiento.

4. **Consumo Energético**: Menor consumo de batería, crítico para dispositivos móviles.

5. **Flexibilidad**: Hiperparámetros ajustables permiten adaptar el modelo a diferentes restricciones de recursos.

6. **Precisión Competitiva**: Mantiene precisión razonable a pesar de las optimizaciones para eficiencia.

#### Limitaciones de MobileNet

1. **Brecha de Precisión**: Aún existe una brecha de precisión comparado con arquitecturas más grandes como ResNet o EfficientNet.

2. **Tareas Complejas**: Puede tener dificultades con tareas que requieren contexto espacial amplio o detalles muy finos.

3. **Transferencia a Nuevos Dominios**: Puede requerir más ajustes para transferir efectivamente a dominios muy diferentes de ImageNet.

4. **Sensibilidad a Hiperparámetros**: El rendimiento puede variar significativamente con diferentes configuraciones de hiperparámetros.

5. **Optimización Específica para Hardware**: El rendimiento óptimo puede requerir implementaciones específicas para diferentes plataformas de hardware.

### Impacto y Legado

El impacto de MobileNet en el campo del aprendizaje profundo ha sido profundo y duradero:

1. **Paradigma de Eficiencia**: Estableció la eficiencia computacional como objetivo primario de diseño, no solo una consideración secundaria.

2. **Métricas de Evaluación**: Popularizó nuevas métricas como precisión por operación o precisión por parámetro para evaluar modelos.

3. **Convoluciones Separables**: Demostró la efectividad de las convoluciones separables en profundidad, ahora ampliamente adoptadas.

4. **Diseño para Dispositivos**: Inspiró una nueva generación de arquitecturas específicamente diseñadas para dispositivos con recursos limitados.

5. **Democratización del Aprendizaje Profundo**: Facilitó el despliegue de modelos CNN en dispositivos accesibles, ampliando el alcance de aplicaciones prácticas.

6. **Búsqueda de Arquitectura**: Pionero en la integración de NAS con objetivos específicos de latencia en hardware real.

Los papers de MobileNet han acumulado decenas de miles de citas, y la arquitectura ha sido implementada en numerosos frameworks y plataformas, convirtiéndose en un estándar de facto para aplicaciones móviles de visión por computadora.

### Conclusión

MobileNet representa un punto de inflexión en el diseño de redes neuronales convolucionales, marcando la transición de arquitecturas centradas exclusivamente en la precisión hacia un paradigma que valora igualmente la eficiencia computacional. Su innovación central —las convoluciones separables en profundidad— junto con sus hiperparámetros ajustables, estableció un nuevo estándar para el despliegue de modelos CNN en dispositivos con recursos limitados.

La evolución de MobileNet a través de sus versiones V1, V2 y V3 demuestra el rápido avance en este campo, incorporando progresivamente técnicas como bloques de cuello de botella invertido, conexiones residuales lineales, búsqueda de arquitectura neural y optimizaciones específicas para hardware.

El impacto de MobileNet trasciende su aplicación directa, habiendo influido significativamente en el diseño de arquitecturas eficientes posteriores y establecido nuevos estándares para la evaluación de modelos en términos de precisión por operación. Su legado perdura en la democratización del aprendizaje profundo, permitiendo que aplicaciones sofisticadas de visión por computadora funcionen directamente en dispositivos de usuario final.

En la próxima lección, exploraremos EfficientNet, una arquitectura que lleva la optimización de recursos a un nuevo nivel mediante un enfoque sistemático de escalado compuesto, logrando un equilibrio aún mejor entre precisión y eficiencia.
