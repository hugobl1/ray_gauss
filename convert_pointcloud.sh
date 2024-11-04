#bash file that converts a pt pointcloud to a ply pointcloud

start_iteration=1000
end_iteration=10000
gap=1000
for i in $(seq $start_iteration $gap $end_iteration)
do
    echo "Converting pointcloud $i"
    python convertpt_to_ply.py -iter $i 
done