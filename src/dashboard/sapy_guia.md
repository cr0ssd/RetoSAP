## Qué es este dashboard
Monitorea comportamiento inusual en los registros (logs) de SAP. La actividad se divide en bloques de tiempo; un modelo de aprendizaje automático marca los bloques atípicos, estos se agrupan en eventos, cada evento relevante se clasifica en un tipo de ataque y un agente forense genera un reporte postmortem de cada uno.

## Controles de la barra lateral
- Idioma / Language: Español o English.
- Periodo: el mes que se analiza.
- Origen de los registros: Sistema SAP o Servicios de IA. Cada origen tiene sus propios indicadores y tipos de ataque posibles.
- Intervalo de análisis: tamaño de los bloques de tiempo (5 min, 15 min, 30 min, 1 h o 3 h). Con intervalos cortos se detectan más eventos breves; con intervalos largos, solo los que se sostienen en el tiempo.
- Recuadro "Periodo seleccionado" (aparece solo si se hizo zoom en la gráfica de actividad) con el botón "Ver todo el mes".
- Contador de reportes forenses del periodo y botón "Actualizar datos en vivo".

## Páginas (menú superior)
1. Resumen. Cinco indicadores (Bloques analizados, Bloques atípicos, Eventos, Eventos relevantes, Tipos de ataque), la gráfica "Actividad y eventos en el tiempo", la gráfica "Eventos relevantes por tipo" (barras por nivel de confianza), la tabla "Eventos de mayor intensidad" (los 5 principales) y una tarjeta "Reportes forenses" con el enlace "Ver reportes".
2. Eventos. El mapa de calor "Aspectos afectados en cada evento" y la tabla "Detalle de eventos" con las pestañas Relevantes y Todos.
3. Tipos de ataque. Explica cómo se asigna el tipo y muestra una tarjeta por cada tipo posible.
4. Postmortems SOC. Los reportes forenses del mes (ver más abajo).
5. Metodología. Los pasos del análisis y dos comprobaciones de calidad.
6. MLOps. Estado del sistema (Health checks) y Entrenamiento en vivo.

## Conceptos clave
- Bloque: tramo de tiempo (según el intervalo de análisis) en el que se miden los indicadores. Los bloques se traslapan a la mitad. Se miden 24 indicadores en Sistema SAP y 18 en Servicios de IA.
- Bloque atípico: bloque cuyo comportamiento se aleja claramente del habitual. Cada semana, el modelo (Isolation Forest) marca el 1 % de bloques más inusuales.
- Evento: grupo de bloques atípicos cercanos en el tiempo (separados por 3 bloques o menos).
- Evento relevante: evento en el que algún indicador se alejó {Z_UMBRAL} desviaciones estándar o más de su promedio semanal. Solo los eventos relevantes se clasifican por tipo y reciben reporte postmortem. Los eventos no relevantes se detectan, pero no se consideran importantes.
- Intensidad (Z): cuántas desviaciones estándar se alejó el indicador más afectado de su promedio semanal. A mayor intensidad, más inusual fue el evento.
- Indicador más afectado: el indicador que más cambió durante el evento. "Principales cambios" resume los 3 que más cambiaron (↑ aumentó, ↓ disminuyó).
- Tipo de ataque: se asigna por votación. Los 3 indicadores que más cambiaron votan por los tipos cuyo patrón coincide con la dirección del cambio (subió o bajó); gana el tipo con más votos.
- Confianza: alta si el tipo ganador supera al segundo por al menos 1 voto; media si lo supera por entre 0.5 y 1; baja si lo supera por menos de 0.5.
- Falla operativa: tipo que no cuenta como ataque (errores y advertencias con caída del tráfico).
- Semanas: las líneas punteadas verticales de la gráfica de actividad marcan el inicio de cada semana; la detección se calcula semana por semana.

## Cómo leer la gráfica "Actividad y eventos en el tiempo" (página Resumen)
- Eje horizontal: tiempo del mes. Eje vertical: número de registros por bloque.
- La línea azul es el volumen de registros por bloque.
- Los puntos rojos son bloques atípicos; al pasar el cursor se ve su evento y tipo.
- Las franjas de color son los eventos relevantes; el color indica el tipo de ataque (la leyenda está arriba de la gráfica).
- Las líneas punteadas verticales marcan el inicio de cada semana.
- Se puede acercar un periodo arrastrando sobre la gráfica o con la barra inferior. Al hacerlo, todo el dashboard (todas las páginas) se recalcula solo para ese periodo. Doble clic en la gráfica, el botón "Ver todo" o "Ver todo el mes" de la barra lateral lo restablecen.
- La gráfica siempre muestra todo el mes; el zoom solo marca el periodo seleccionado.

## Cómo leer el mapa de calor "Aspectos afectados en cada evento" (página Eventos)
Cada fila es un evento relevante y cada columna, un grupo de indicadores. El número y el tono muestran cuánto cambió ese grupo durante el evento comparado con una semana normal; los tonos más oscuros indican un cambio mayor.

## Tabla "Detalle de eventos" (página Eventos)
Ordenada por intensidad. Columnas: Evento, Tipo, Confianza, Inicio, Fin, Intensidad, Indicador más afectado y Principales cambios.

## Postmortems SOC: dónde descargar y qué contiene
- Descargar el reporte en PDF: página Postmortems SOC, arriba a la derecha, botón "Descargar PDF". Genera un PDF ejecutivo de varias páginas con el resumen del mes y el detalle de cada incidente. Incluye los reportes que se estén mostrando (según el periodo seleccionado y los filtros activos).
- Botón "Descargar Markdown": baja el consolidado completo del mes en Markdown.
- Botón "Enviar por correo": abre un formulario para enviar ese PDF por correo electrónico.
- No hay un PDF por cada incidente individual; el PDF es el consolidado.
- Filtros de la página: Origen, Tipo de ataque y un buscador (reporte, táctica MITRE, aplicación o IP).
- Indicadores de la página: Incidentes del mes, Tácticas MITRE y Aplicaciones SAP (con las IP involucradas).
- Los incidentes se pueden ver en dos pestañas: "Por campaña / familia de amenaza" y "Cronología completa de incidentes".
- Cada incidente tiene un resumen ejecutivo y, en un desplegable, el "Reporte completo" y la "Evidencia" (aplicaciones SAP afectadas, IPs involucradas, muestra de registros y metadatos).
- Si un periodo no tiene reportes todavía, aparecerán cuando el agente forense procese los eventos relevantes de ese periodo.

## Metodología (página Metodología)
1. División en bloques de 5 min, 15 min, 30 min, 1 h y 3 h.
2. Detección de lo atípico con Isolation Forest (marca el 1 % de bloques más inusuales de cada semana).
3. Agrupación en eventos (bloques atípicos separados por 3 bloques o menos).
4. Clasificación por reglas fijas que comparan los 3 indicadores que más cambiaron con el patrón de cada tipo de ataque.
5. Comprobación: un agrupamiento automático (k-means) de los bloques atípicos se compara con la clasificación. La coincidencia se mide con dos índices, ARI y NMI (1 es coincidencia total, 0 es azar). No es una validación independiente porque ambos métodos usan los mismos indicadores.
La página también muestra "Resultados según el intervalo de análisis" (eventos y eventos relevantes por intervalo).

## MLOps (página MLOps)
- Health checks: estado del sistema en grupos como Infraestructura, Datos, Modelo y Agente forense.
- Entrenamiento en vivo: corre el mismo pipeline de producción sobre una semana elegida, en tablas temporales de SAP HANA, sin modificar los resultados guardados. Se elige tabla de logs, semana, intervalo y origen, y se pulsa "Iniciar entrenamiento". Intervalos cortos tardan más (1 h suele tardar menos de un minuto; 5 min puede tardar varios minutos).
