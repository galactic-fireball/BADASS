The host model is used as a simplified placeholder in the event that the stellar continuum isn't of any interest.  These are single-stellar population templates from the EMILES library, and do not have a low enough resolution for reliable stellar LOSVD fitting.

```python
host_options = {
  "age"       : [1.0,5.0,10.0], # Ages to include in Gyr; [0.09 Gyr - 14 Gyr] 
  "vel_const" : {"bool":False, "val":0.0}, # hold velocity constant?
  "disp_const": {"bool":False, "val":150.0} # hold dispersion constant?
}
```