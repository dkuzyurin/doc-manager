# Определение вида договора с интерпретацией результатов

Реализация решения задачи определения типа договора в рамках хакатона [X-MAS HACK 2022](https://www.zavodit.ru/ru/calendar/event/21)

## Описание задачи

Разработать решение для автоматического определения вида договора. Решение должно принимать на вход документ в форматах doc, docx, pdf и выдавать вид договора, а также интерпретировать результаты. Интерпретация результатов предполагает наличие признаков и критериев, по которым был выбран вид договора. Успех решения будет определяться не только по тому, насколько правильно определяется вид договора, но и по качеству интерпретации результатов.
 
## Реализованный функционал

1. Загрузка файла с договором (DOC, DOCX, PDF, RTF)
2. Определение вероятностей принадлежности договора к каждому из известных типов
3. Определение степени похожести на договоры наиболее вероятного типа
4. Возможность отправить договор отвественному за договоры данного типа (при высокой уверенности модели в ответе), либо на дополнительную проверку (в случае низкой уверенности модели)
5. Вывод фрагментов текста договора, наиболее сильно повлиявших на принятое моделью решение
6. Вывод списка всех проверенных ранее договоров с возможностью поиска (по имени файла, по дате, по типу договора)
7. Вывод детальной информации по выбранному из списка ранее проверенному договору

## Видео-демонстрация основного функционала

https://www.youtube.com/watch?v=Hgl4kudirAU