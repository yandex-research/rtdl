{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree: api
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
      :toctree: api
   {% for item in methods %}
      ~{{ name }}.{{ item }}

   {%- endfor %}
   {% for item in ['__call__', '__len__', '__getitem__', '__setitem__', '__iter__', '__enter__', '__exit__', '__getstate__', '__setstate__',] %}
      {% if item in members %}
      ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endblock %}
