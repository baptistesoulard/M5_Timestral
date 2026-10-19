## M5

| configuration | WRMSSE median (bottom-up) | WRMSSE mean (bottom-up) | WSPL | ms/series |
|---|---|---|---|---|
| t5m_calendar | 1.8337 | 0.7121 | 0.1650 | 91 |
| t28m_none | 1.7143 | 0.6729 | 0.1689 | 50 |
| t28m_calendar | 1.7258 | 0.6150 | 0.1620 | 351 |
| t28m_full | 1.7274 | 0.6272 | 0.1618 | 525 |
| t28m_rich | 1.7537 | 0.6601 | 0.1675 | 1080 |
| t209m_none | 1.7240 | 0.6541 | 0.1647 | 269 |
| t209m_calendar | 1.7678 | 0.6168 | 0.1540 | 2441 |
| t0g_none | 1.7094 | 0.6389 | 0.1702 | 202 |
| t0g_calendar | 1.7112 | 0.5958 | 0.1612 | 1042 |
| arima | | 0.8688 | | |

## VN1

| method | VN1 score | MAE% | Bias% |
|---|---|---|---|
| vn1_t0g_none (P57, chosen on 4 backtests) | 0.4479 | 0.4472 | +0.0008 |
| vn1_1st | 0.4637 | 0.4525 | +0.0112 |
| vn1_2nd | 0.4657 | 0.4626 | -0.0031 |
| vn1_t209m_none (P57, chosen on 4 backtests) | 0.4667 | 0.4475 | +0.0193 |
| vn1_t5m_none (P57, chosen on 4 backtests) | 0.4675 | 0.4507 | +0.0168 |
| vn1_t28m_none (P57, chosen on 4 backtests) | 0.4701 | 0.4506 | +0.0195 |
| vn1_t209m_price (P57, chosen on 4 backtests) | 0.4713 | 0.4504 | +0.0210 |
| vn1_3rd | 0.4758 | 0.4600 | +0.0158 |
| vn1_4th | 0.4774 | 0.4754 | -0.0020 |
| vn1_t28m_price (P57, chosen on 4 backtests) | 0.4796 | 0.4531 | +0.0266 |
| vn1_5th | 0.4808 | 0.4499 | +0.0309 |
| vn1_t28m_price (median) | 0.4941 | 0.4297 | -0.0644 |
| vn1_t0g_calendar (P54, chosen on 4 backtests) | 0.4955 | 0.4503 | -0.0451 |
| vn1_t0g_calendar (P54, chosen on phase 1 only) | 0.4955 | 0.4503 | -0.0451 |
| vn1_t5m_none (median) | 0.4957 | 0.4281 | -0.0676 |
| vn1_t28m_none (median) | 0.4979 | 0.4278 | -0.0701 |
| vn1_t209m_none (median) | 0.4989 | 0.4246 | -0.0743 |
| vn1_t209m_price (median) | 0.4998 | 0.4273 | -0.0725 |
| vn1_t209m_none (P59, chosen on phase 1 only) | 0.5048 | 0.4572 | +0.0476 |
| naive | 0.5070 | 0.5013 | -0.0056 |
| vn1_t209m_sidx (P56, chosen on 4 backtests) | 0.5087 | 0.4421 | -0.0666 |
| vn1_t209m_price (P59, chosen on phase 1 only) | 0.5094 | 0.4601 | +0.0493 |
| vn1_t28m_sidx (P56, chosen on 4 backtests) | 0.5114 | 0.4447 | -0.0667 |
| vn1_t0g_none (median) | 0.5240 | 0.4292 | -0.0948 |
| vn1_t28m_none (P60, chosen on phase 1 only) | 0.5301 | 0.4663 | +0.0638 |
| vn1_t0g_calendar (median) | 0.5392 | 0.4425 | -0.0967 |
| vn1_t28m_price (P60, chosen on phase 1 only) | 0.5407 | 0.4691 | +0.0716 |
| vn1_t0g_none (P62, chosen on phase 1 only) | 0.5494 | 0.4722 | +0.0772 |
| vn1_t5m_none (P61, chosen on phase 1 only) | 0.5510 | 0.4731 | +0.0780 |
| ses | 0.5672 | 0.5139 | +0.0533 |
| vn1_t28m_sidx (mean) | 0.5765 | 0.5125 | +0.0639 |
| vn1_t28m_sidx (median) | 0.5783 | 0.4376 | -0.1407 |
| vn1_t209m_sidx (median) | 0.5790 | 0.4354 | -0.1436 |
| vn1_t28m_calendar (median) | 0.5800 | 0.5665 | -0.0135 |
| vn1_t209m_sidx (mean) | 0.5835 | 0.5154 | +0.0681 |
| vn1_t28m_calendar (P48, chosen on phase 1 only) | 0.5966 | 0.5627 | -0.0339 |
| arima | 0.6114 | 0.5276 | +0.0838 |
| vn1_t0g_calendar (mean) | 0.6299 | 0.5306 | +0.0993 |
| vn1_t5m_none (mean) | 0.6443 | 0.5189 | +0.1254 |
| ets | 0.6449 | 0.5446 | +0.1004 |
| vn1_t209m_sidx (P44, chosen on phase 1 only) | 0.6541 | 0.4390 | -0.2151 |
| vn1_t28m_sidx (P44, chosen on phase 1 only) | 0.6543 | 0.4414 | -0.2128 |
| vn1_t28m_calendar (mean) | 0.6748 | 0.6048 | +0.0701 |
| vn1_t28m_none (mean) | 0.6938 | 0.5353 | +0.1585 |
| vn1_t209m_none (mean) | 0.7001 | 0.5377 | +0.1624 |
| vn1_t28m_calendar (P61, chosen on 4 backtests) | 0.7074 | 0.6042 | +0.1032 |
| vn1_t209m_price (mean) | 0.7127 | 0.5450 | +0.1676 |
| vn1_t0g_none (mean) | 0.7152 | 0.5512 | +0.1640 |
| vn1_t28m_price (mean) | 0.7157 | 0.5458 | +0.1699 |
| imapa | 0.7568 | 0.5980 | +0.1587 |
| adida | 0.7573 | 0.5986 | +0.1587 |
| tsb | 0.7581 | 0.5827 | +0.1753 |
| snaive | 0.7852 | 0.7762 | -0.0090 |
| ma12 | 0.8046 | 0.6097 | +0.1949 |
| sba | 0.8637 | 0.6530 | +0.2107 |
