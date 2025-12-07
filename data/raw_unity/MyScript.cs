using UnityEngine;

public class MyScript : MonoBehaviour
{
    public float moveSpeed = 5f;
    public Rigidbody playerRb;
    public Transform target;
    private bool isStunned = false;
    private float stunTimer = 0f;

    void Start()
    {
        if (playerRb == null)
        {
            playerRb = GetComponent<Rigidbody>();
        }
    }

    void Update()
    {
        if (isStunned)
        {
            HandleStunTimer();
        }
        else
        {
            HandleMovement();
        }
    }

    void HandleMovement()
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        Vector3 direction = new Vector3(horizontal, 0f, vertical);

        if (direction.magnitude > 0.1f)
        {
            playerRb.MovePosition(transform.position + direction.normalized * moveSpeed * Time.deltaTime);
        }
    }

    public void ApplyStun(float duration)
    {
        isStunned = true;
        stunTimer = duration;
    }

    void HandleStunTimer()
    {
        stunTimer -= Time.deltaTime;

        if (stunTimer <= 0f)
        {
            isStunned = false;
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Enemy"))
        {
            ApplyStun(1.5f);
        }
    }

    public void MoveTowardTarget()
    {
        if (target == null) return;

        Vector3 dir = (target.position - transform.position).normalized;
        playerRb.MovePosition(transform.position + dir * moveSpeed * Time.deltaTime);
    }
}
