


============Build application====================
docker compose -f lims.yml --build


==============start application==================
docker compose -f lims.yml up -d


=================Access the Dashboard============
http://localhost:5000



==============View logs===========================
docker compose -f lims.yml logs -f

===============rebuild after changes==============
docker compose -f lims.yml up -d --build

==============Stop application====================
docker compose -f lims.yml down


